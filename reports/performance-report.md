# Rapport de performance — Home Credit Scoring API
# Étape 4 : Analyse et optimisation des performances

## Contexte et objectif

L'objectif de cette étape est d'analyser les performances de l'API de scoring
et d'identifier les goulots d'étranglement, puis d'appliquer des optimisations
ciblées et d'en mesurer le gain.

**Méthodologie :**
- Benchmark HTTP (300 requêtes, 20 warmup) via `bench_api.py`
- Chronomètres par étape dans `/predict` (hash, preprocessing, stats, inference, logging)
- Profiling CPU via `cProfile` (1000 appels, hors réseau)

Commande :

```
uv run python monitoring/bench_api.py --n 300
```
---

## 2. Baseline — avant optimisation

### 2.1 Latence client (réseau inclus)

| Métrique | Valeur |
|---|---|
| mean | 14.85 ms |
| p50  | 14.36 ms |
| p95  | 17.95 ms |
| max  | 38.94 ms |

### 2.2 Timings par étape (côté serveur)

| Étape | mean | p95 | % du total |
|---|---|---|---|
| `hash_ms` | 0.344 ms | 0.408 ms | 2% |
| `preprocessing_ms` | 5.772 ms | 7.056 ms | **39%** |
| `stats_ms` | 0.638 ms | 0.801 ms | 4% |
| `inference_ms` | 5.883 ms | 7.378 ms | **40%** |

**Diagnostic :** deux goulots quasi-égaux — preprocessing et inférence,
~5.8ms chacun. Le preprocessing (construction DataFrame pandas et l'alignement des 512 colonnes)
coûte autant que XGBoost lui-même.

---

## 3. Analyse cProfile

Profiling hors réseau sur 1000 appels (`profiling/profile_inference.py`).

Le profiling `cProfile` confirme que la création/alignement DataFrame domine un epartie importante du temps CPU.

### Version slow (dict → DataFrame)
- **Temps total 1000 appels : 31.0s — soit 31.0ms/appel**
- `make_df_slow` : 13.7ms/appel consommés en construction DataFrame
- Cause identifiée : construction d'un DataFrame pandas avec 512 colonnes

### Version fast (numpy → DataFrame)
- **Temps total 1000 appels : 17.6s — soit 17.6ms/appel**
- `make_df_fast` : 0.83ms/appel — division par ~16

Les rapports détaillés sont disponibles dans :
- `reports/profiling/cprofile_before.txt`
- `reports/profiling/cprofile_after.txt`

---

## 4. Optimisation appliquée

**Remplacement de la construction DataFrame par dict → numpy array + DataFrame**

1. Remplir un tableau NumPy (1, n_features) en float32 via mapping feature->index
2. Wrapper en `pandas.DataFrame(x, columns=feature_cols)`uniquement à la fin (nécessaire car le pipeline dépend des noms de colonnes)

```python
# AVANT — dict à 512 clés puis DataFrame
row = {c: payload.get(c, None) for c in feature_cols}  # 512 lookups
df = pd.DataFrame([row], columns=feature_cols)

# APRÈS — numpy array float32 puis DataFrame
x = np.full((1, len(feature_cols)), np.nan, dtype=np.float32)
for k, v in payload.items():       # itère seulement les features envoyées
    x[0, feature_index[k]] = fv    # lookup O(1) via dict précalculé
df = pd.DataFrame(x, columns=feature_cols)
```

**Gains supplémentaires :**
- réduire le temps de preprocessing

- garder la compatibilité avec le pipeline sklearn (sélection par noms)

---

## 5. Résultats après optimisation

### 5.1 Latence client

| Métrique | Avant | Après | Gain |
|---|---|---|---|
| mean | 14.85 ms | 8.09 ms | **-46%** |
| p50  | 14.36 ms | 7.61 ms | **-47%** |
| p95  | 17.95 ms | 9.94 ms | **-45%** |
| max  | 38.94 ms | 32.23 ms | -17% |

### 5.2 Timings par étape

| Étape | Avant mean | Après mean | Gain | Avant p95 | Après p95 |
|---|---|---|---|---|---|
| `hash_ms` | 0.344 ms | 0.315 ms | -8% | 0.408 ms | 0.400 ms |
| `preprocessing_ms` | 5.772 ms | 0.514 ms | **-91%**   | 7.056 ms | 0.642 ms |
| `stats_ms` | 0.638 ms | 0.029 ms | **-95%**  | 0.801 ms | 0.037 ms |
| `inference_ms` | 5.883 ms | 5.242 ms | -11% | 7.378 ms | 6.407 ms |

Le profiling cProfile après optimisation montre une baisse forte du temps passé dans la préparation des données, avec un déplacement du coût vers predict_proba (inférence).

---

## 6. Validation & non-régression

Tests automatisés pytest : /health, /predict, rejet features inconnues (422), rejet payload vide.

Contrat API inchangé (mêmes endpoints, même format de réponse).

Les features manquantes restent supportées (remplies en NaN).

## 7. Décision et limites

**Piste explorée :** ONNX Runtime : non retenu dans cette version (conversion pipeline + risque de divergence / effort non rentable vs gains déjà obtenus)

**Conclusion : non retenu**, pour les raisons suivantes :

- Le pipeline contient un `ColumnTransformer` avec des transformers nommés
  qui dépendent des noms de colonnes pandas. La conversion complète
  pipeline → ONNX nécessite une refactorisation ou une conversion complexe
  via `skl2onnx` avec type mapping manuel pour chaque transformer.
- Le bottleneck principal (preprocessing) a déjà été éliminé à -91%.
- L'inférence résiduelle (~5.2ms mean, 6.4ms p95) est stable et acceptable.
- **Règle d'or appliquée : effort vs gain.** L'effort de conversion ONNX
  est élevé (risque de régression, validation des prédictions, maintenance)
  pour un gain estimé à ~2-3ms sur un p95 déjà à 9.94ms.

---

## 7. Conclusion

| Objectif | Cible | Résultat |
|---|---|---|
| Réduire preprocessing | < 1.5ms | **0.51ms**  |
| p95 latence client | < 10ms | **9.94ms**  |
| Documenter avec cProfile | V | V |
| Évaluer ONNX | V | Décision motivée V |

Le goulot d'étranglement restant est l'inférence XGBoost (~5.2ms),
inhérente au modèle et non optimisable sans changer l'architecture.
Le ratio performance/complexité ne justifie pas d'aller plus loin
sur cette version du pipeline.
