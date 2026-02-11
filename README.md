# Home Credit — MLOps Scoring API (FastAPI + Docker + CI/CD)

Ce dépôt contient une solution MLOps de bout en bout pour un modèle de scoring crédit :
- **tracking & registry MLflow** (expérimentations + modèle final versionné),
- **API FastAPI** exposant une prédiction de probabilité de défaut,
- **tests automatisés (pytest)**,
- **conteneurisation Docker**,
- **CI/CD GitHub Actions** (tests + build Docker + déploiement sur Hugging Face Spaces),
- (à venir ) **monitoring & data drift**.

---

## Sommaire
- [Architecture](#architecture)
- [Pré-requis](#pré-requis)
- [Installation (local)](#installation-local)
- [Lancer l’API (local)](#lancer-lapi-local)
- [Utiliser l’API](#utiliser-lapi)
- [Tests](#tests)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Artefacts du modèle](#artefacts-du-modèle)
- [Conventions & décisions](#conventions--décisions)
- [À faire / Roadmap](#à-faire--roadmap)

---

## Architecture
.
├── src/
│ └── app/
│ ├── main.py 
│ ├── schemas.py 
│ └── artifacts/
│ ├── model.pkl
│ ├── feature_cols.json
│ └── threshold_config.json
├── tests/
│ ├── test_health.py
│ ├── test_predict.py
│ └── sample_payload.json
├── notebooks/ 
├── Dockerfile
├── pyproject.toml 
├── uv.lock
└── .github/workflows/ci-cd.yml 

---

## Pré-requis
- Python **3.12** (recommandé, aligné avec l’image Docker)
- `uv` (gestion d’environnement / dépendances)
- (optionnel) Docker Desktop

---

## Installation (local)

```bash
# à la racine du repo
uv sync --frozen --dev
```

---

## Lancer lÀPI
```
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Puis ouvrir :

- Swagger UI : http://127.0.0.1:8000/docs

- Healthcheck : http://127.0.0.1:8000/health

---

## Utiliser l'API

Endpoints
- GET /health : statut + modèle chargé

- GET /metadata : informations modèle + seuil

- POST /predict : prédiction d’un client

---

## Tests

```
uv run pytest -q
```

Tests couverts :

- /health répond 200

- /predict avec payload valide répond 200

- rejet si feature inconnue (422)

- rejet si features vide (422)

---

## Docker

Build
```
docker build -t homecredit-api .
```

Run
```
docker run --rm -p 8000:8000 homecredit-api
```

http://127.0.0.1:8000/docs

---

## CI/CD

Le pipeline GitHub Actions exécute :

1. Tests (pytest)

2. Build Docker 

3. Déploiement sur Hugging Face Spaces 

Fichier : .github/workflows/ci-cd.yml

**Secrets / Variables nécessaires**

Dans GitHub → Settings → Secrets and variables :

**Secret**
HF_TOKEN : token Hugging Face avec accès à la Space

**Variables**

HF_USERNAME : username HF

HF_SPACE_NAME : nom de la Space (repo Spaces)

---

## Artefacts du modèle

Le runtime de l’API repose sur :

`src/app/artifacts/model.pkl` : pipeline entraîné (préprocessing + modèle)

`src/app/artifacts/feature_cols.json` : liste des features attendues (ordre training)

`src/app/artifacts/threshold_config.json` : seuil + coût métier

Le modèle est chargé une seule fois au démarrage de l’API, puis réutilisé à chaque requête.

---

## Conventions & décisions

- Problème déséquilibré : gestion via scale_pos_weight (XGBoost) + seuil métier.

- Métrique principale : business cost avec FN=10, FP=1.

- Seuil : optimisé sur OOF (Out-Of-Fold) pour minimiser le coût.

- API : validation stricte des features (rejette les inconnues, complète les manquantes par None/NaN).

---

##À faire

 Stockage des logs de production (inputs/outputs/latence)

 Dashboard de monitoring (distribution scores, latence, taux erreur)

 Détection de data drift (PSI / KS) sur features clés + alerting

 Endpoint batch /predict_batch

 Sécurité minimale (API key / rate limit)
