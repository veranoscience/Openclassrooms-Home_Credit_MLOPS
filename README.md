---
title: Home Credit — MLOps Scoring API
sdk: docker
app_port: 8000
emoji: 🚀
colorFrom: blue
colorTo: red
license: mit
pinned: false
tags:
  - mlops
  - fastapi
  - docker
  - xgboost
  - credit-scoring
  - monitoring
  - mlflow
---
# Home Credit — MLOps Scoring API (FastAPI + Docker + CI/CD)

Sommaire:
- **tracking & registry MLflow** (expérimentations + modèle final versionné),
- **API FastAPI** exposant une prédiction de probabilité de défaut,
- **tests automatisés (pytest)**,
- **conteneurisation Docker**,
- **CI/CD GitHub Actions** (tests + build Docker + déploiement sur Hugging Face Spaces),
- **Monitoring (logs JSONL + Postgres) & data drift (PSI / Evidently)**,
- **Profiling cProfile + optimisation preprocessing (NumPy → DataFrame)**


---

## Sommaire
- [Architecture](#architecture)
- [Pré-requis](#pré-requis)
- [Installation (local)](#installation-local)
- [Lancer l’API (local)](#lancer-lapi-local)
- [Utiliser l’API](#utiliser-lapi)
- [Démo HF](#demo_hf)
- [Tests](#tests)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Artefacts du modèle](#artefacts-du-modèle)
- [Conventions & décisions](#conventions--décisions)
- [Optimisation performance](#optimisation)


---

## Architecture

```text
.
├── src/
├── tests/
├── notebooks/ 
├── monitoring/
├── prod_logs
├── Dockerfile
├── docker-compose.monitoring.yml
├── pyproject.toml 
├── uv.lock
├── .github/workflows/
├── README.md
├── reports
├── requirements.txt

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
## Démo HF

Base URL : `https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space`

Exemples :

```
curl -s https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space/health
curl -s https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space/metadata

curl -s -X POST https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space/predict \
  -H "Content-Type: application/json" \
  --data @tests/sample_payload.json
```

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

## Optimisation performance

```md

Avant optimisation :
- mean ~ 13.76 ms, p95 ~ 15.99 ms
- preprocessing p95 ~ 6.28 ms

Après optimisation preprocessing (NumPy → DataFrame) :
- mean ~ 8.09 ms, p95 ~ 9.94 ms
- preprocessing p95 ~ 0.64 ms
- inference p95 ~ 6.41 ms (bottleneck restant)




