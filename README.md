# Rossmann Sales Forecasting

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-blue)](https://github.com/microsoft/LightGBM)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-orange)](https://catboost.ai/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-deployable-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Multi-horizon **sales forecasting** for the [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) Kaggle competition.

A clean, production-shaped monorepo: **LightGBM / CatBoost** models behind a **FastAPI** service, a separate **static UI**, and one-time **PostgreSQL migration** — everything Dockerized and CI-ready for **Yandex Cloud**.

> **RU** — многогоризонтное прогнозирование продаж (Rossmann Store Sales): ML-пайплайн на LightGBM/CatBoost, REST API на FastAPI, отдельный веб-интерфейс и Postgres-хранилище истории. Готов к деплою в Yandex Cloud.

---

## Highlights

- **Multi-horizon forecasting** — 1/2/4/6/12-week aggregation with iterative multi-step prediction.
- **API / UI split** — the frontend is a plain static site that talks to the API over REST. No server-side rendering, no coupled monolith.
- **Database-backed history** — prediction context (past sales) is read from PostgreSQL, not baked into the image.
- **One-time DB migration** — `scripts/migrate_db.py` loads historical data once. No per-commit DB image copying, no wasted disk.
- **Docker / k8s ready** — two small images (`api`, `ui`), built with **kaniko** (no Docker daemon required in CI).

---

## Architecture

```
                ┌──────────────────────────┐
                │  ui  (nginx, static)     │
                │  index.html / config.js  │
                └────────────┬─────────────┘
                             │  POST /predict (xlsx)
                             ▼
                ┌──────────────────────────┐
                │  api  (FastAPI :8000)    │
                │  /health, /predict       │
                └───────┬──────────────────┘
                        │  SELECT history ...
                        ▼
                ┌──────────────────────────┐
                │  db  (PostgreSQL)        │
                │  table: history          │
                └──────────────────────────┘
```

| Component | Path | Stack | Purpose |
|-----------|------|-------|---------|
| **api** | `api/` | FastAPI, LightGBM, SQLAlchemy | Forecasting endpoint, loads artifacts + history |
| **ui** | `ui/` | HTML, Bootstrap, SheetJS | Static page, sends Excel → `/predict`, downloads result |
| **db** | — | PostgreSQL 17 | Historical sales data (`history` table) |
| **scripts** | `scripts/` | Python, psycopg2 | One-time DB migration |

---

## API

Single forecasting endpoint:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe (`{"status":"ok"}`) |
| `/predict` | POST | Accepts `.xlsx` file (multipart field `data`), returns `.xlsx` with predictions |

Swagger UI is available at `/docs`.

### Example

```bash
curl -X POST \
  -F "data=@input.xlsx" \
  -o predictions.xlsx \
  http://localhost:8000/predict
```

**Input columns** (the Rossmann schema): `Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval`.

---

## Quick Start

### 1. Local development (Docker Compose)

```bash
docker compose up --build
```

- **UI**: http://localhost:8080
- **API**: http://localhost:8000 (`/docs` for Swagger)

### 2. Bare Python (no Docker)

```bash
pip install uv
uv sync --group deploy
cd api
uv run uvicorn main:app --reload
```

### 3. Seed the database (one time)

Point `DATABASE_URL` at your Postgres and load `history.csv`:

```bash
uv run python scripts/migrate_db.py
```

Requires env: `DATABASE_URL` (and `HISTORY_CSV`, default `history.csv`).

---

## Deployment: Yandex Cloud

The CI pipeline (**GitLab CI**, runner on **Kubernetes**) builds and pushes both images to **Yandex Container Registry** using **kaniko** — no Docker-in-Docker, no daemon socket, no `privileged` mode.

### Pipeline stages

| Job | Image | Push target |
|-----|-------|-------------|
| `build-api` | `Dockerfile.api` | `cr.yandex/<REGISTRY_ID>/ts-api` |
| `build-ui` | `Dockerfile.ui` | `cr.yandex/<REGISTRY_ID>/ts-ui` |

### CI/CD variables (GitLab → Settings → CI/CD → Variables)

| Variable | Type | Description |
|----------|------|-------------|
| `REGISTRY_ID` | Variable | Your Yandex Container Registry ID (e.g. `crp...`) |
| `YC_SA_KEY` | Variable | Service account **authorized key** (JSON) with role `container-registry.images.pusher` |

> **Why not OAuth tokens?** As of 2026, Yandex rejects OAuth tokens issued after `2026-06-01` for registry/`json_key` auth. A **service account authorized key** (`json_key`) is the supported, non-expiring way to authenticate in CI.

### One-time DB migration to the cloud

```bash
# dump schema+data on your machine
pg_dump "postgresql://user:pass@localhost:5433/db" > history_dump.sql

# restore into the managed/k8s Postgres
psql "postgresql://user:pass@<cloud-db>:6432/db" < history_dump.sql
```

Or, if you keep `history.csv` around, just run `scripts/migrate_db.py` against the cloud DB once. No DB image is ever copied by CI.

---

## Project Structure

```
.
├── api/                      # FastAPI service
│   ├── main.py               # /health, /predict
│   ├── pipeline.py           # feature engineering, preprocessing, prediction
│   └── best_model/           # serialized model artifacts
├── ui/                       # static frontend (served by nginx)
│   ├── index.html            # spreadsheet editor + upload/download
│   ├── config.js             # API base URL
│   └── static/               # styles, sample workbook
├── scripts/
│   └── migrate_db.py         # one-time DB load from history.csv
├── tests/                    # smoke tests
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml        # api + ui + db
├── .gitlab-ci.yml            # kaniko build & push
├── pyproject.toml / uv.lock
└── README.md
```

---

## ML Pipeline

```
Upload Excel → Validate columns & data
  → Feature engineering (lags, rolling stats, percent changes)
  → Preprocessing (scaling, imputation)
  → Weekly aggregation & bucketing
  → Iterative multi-step forecasting (LightGBM / CatBoost)
  → Return predictions Excel
```

Context is pulled from the DB (`history`), so the model uses up-to-date past sales for each store.

### Experiments

All configurations trained **with** and **without** IQR outlier removal:

| Aggregation | Forecast Steps | LightGBM | CatBoost |
|-------------|---------------|----------|----------|
| 1 week      | 12 steps      | yes      | yes      |
| 2 weeks     | 6 steps       | yes      | yes      |
| 4 weeks     | 3 steps       | yes      | yes      |
| 6 weeks     | 2 steps       | yes      | yes      |
| 12 weeks    | 1 step        | yes      | yes      |

Results and SHAP plots are logged to **MLflow**.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Models | LightGBM, CatBoost |
| API | FastAPI, Uvicorn |
| UI | HTML, Bootstrap, SheetJS, nginx |
| Data | Pandas, PostgreSQL, openpyxl, SQLAlchemy |
| ML | Scikit-learn, SHAP, Optuna, MLflow |
| Build / CI | Docker, kaniko, GitLab CI |
| Deploy | Yandex Cloud, Kubernetes |

---

## Tests

```bash
uv run python -m pytest tests/
```

Covers artifact loading, column/data validation, and app routes.

---

## GitHub repository keywords & description

To make the repo discoverable, set these in **GitHub → Settings → General → Topics** (and in the repo description field):

**Keywords (Topics):** `machine-learning`, `forecasting`, `time-series`, `lightgbm`, `catboost`, `fastapi`, `python`, `kaggle`, `rossmann`, `sales-forecasting`, `docker`, `kubernetes`, `gitlab-ci`, `yandex-cloud`

**Short description:** `Multi-horizon sales forecasting (Rossmann) — LightGBM/CatBoost + FastAPI + static UI + PostgreSQL. Docker & k8s ready, CI for Yandex Cloud.`

---

## License

MIT — see [LICENSE](LICENSE).
