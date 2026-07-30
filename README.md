# Rossmann Sales Forecasting

Multi-horizon sales forecasting API for [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset. LightGBM and CatBoost models with a simple web UI for uploading Excel files and downloading predictions.

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140-green)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-orange)]()

---

## Overview

The project includes:
- **ML Pipeline** — feature engineering, preprocessing, iterative multi-step forecasting with LightGBM and CatBoost
- **REST API** — FastAPI endpoint for predictions
- **Web UI** — simple Bootstrap page for uploading `.xlsx` files and downloading forecasts
- **Docker** — single-container deployment

## API

Single endpoint: `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `data` | file | Excel file (.xlsx) with required columns |

Returns: `.xlsx` file with forecasted sales.

## Quick Start

### Local

```bash
pip install uv
uv sync
uv run uvicorn main:app --reload
```

Open `http://localhost:8000` — upload an Excel file and get predictions.

### Docker

```bash
docker build -t rossmann-api .
docker run -p 8000:8000 rossmann-api
```

## Project Structure

```
├── main.py              # FastAPI app + UI routes
├── pipeline.py          # ML pipeline (feature engineering, prediction)
├── templates/
│   └── index.html       # Web UI (Bootstrap)
├── static/
│   └── style.css        # Custom styles
├── best_model/          # Serialized model artifacts
├── Dockerfile
└── pyproject.toml
```

## ML Pipeline

```
Upload Excel → Validate columns → Load historical context
  → Feature engineering (lags, rolling stats, percent changes)
  → Preprocessing (scaling, imputation)
  → Weekly aggregation & bucketing
  → Iterative multi-step forecasting (LightGBM / CatBoost)
  → Return predictions Excel
```

## Experiments

All configurations trained **with** and **without** IQR outlier removal:

| Aggregation | Forecast Steps | LightGBM | CatBoost |
|-------------|---------------|----------|----------|
| 1 week      | 12 steps      | yes      | yes      |
| 2 weeks     | 6 steps       | yes      | yes      |
| 4 weeks     | 3 steps       | yes      | yes      |
| 6 weeks     | 2 steps       | yes      | yes      |
| 12 weeks    | 1 step        | yes      | yes      |

Results and SHAP plots are logged to MLflow.

## Tech Stack

| Category | Tools |
|----------|-------|
| **Models** | LightGBM, CatBoost |
| **API** | FastAPI, Uvicorn |
| **UI** | Bootstrap, Jinja2 |
| **Data** | Pandas, openpyxl |
| **ML** | Scikit-learn, SHAP, Optuna, MLflow |
| **Deploy** | Docker |

---

## RU

# Прогнозирование продаж Rossmann

API для многогоризонтного прогнозирования продаж для датасета [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales). Модели LightGBM и CatBoost с простым веб-интерфейсом для загрузки Excel-файлов и скачивания прогнозов.

### Обзор

- **ML-пайплайн** — инжиниринг признаков, предобработка, итеративное многошаговое прогнозирование (LightGBM, CatBoost)
- **REST API** — эндпоинт FastAPI для предсказаний
- **Веб-интерфейс** — простая страница на Bootstrap для загрузки `.xlsx` и скачивания результатов
- **Docker** — деплой в одном контейнере

### Быстрый старт

**Локально:**

```bash
pip install uv
uv sync
uv run uvicorn main:app --reload
```

Откройте `http://localhost:8000` — загрузите Excel-файл и получите прогноз.

**Docker:**

```bash
docker build -t rossmann-api .
docker run -p 8000:8000 rossmann-api
```

### Структура проекта

```
├── main.py              # FastAPI + UI роуты
├── pipeline.py          # ML-пайплайн
├── templates/
│   └── index.html       # Веб-интерфейс (Bootstrap)
├── static/
│   └── style.css        # Стили
├── best_model/          # Модельные артефакты
├── Dockerfile
└── pyproject.toml
```

### Технологический стек

| Категория | Инструменты |
|-----------|-------------|
| **Модели** | LightGBM, CatBoost |
| **API** | FastAPI, Uvicorn |
| **UI** | Bootstrap, Jinja2 |
| **Данные** | Pandas, openpyxl |
| **ML** | Scikit-learn, SHAP, Optuna, MLflow |
| **Деплой** | Docker |
