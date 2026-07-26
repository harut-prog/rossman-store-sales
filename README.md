
# 🏪 Rossmann Sales Forecasting

Multi-horizon sales forecasting pipeline for [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset using **LightGBM** and **CatBoost** with **Optuna** hyperparameter tuning, **MLflow** experiment tracking, and **SHAP** model interpretation.

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-orange)]()
[![MLflow](https://img.shields.io/badge/MLflow-3.14-blue)]()
[![Optuna](https://img.shields.io/badge/Optuna-4.9-purple)]()

---

**EN** | [RU](#-ru)

---

## Key Features

- **Multi-horizon forecasting** — models trained at 5 aggregation levels: 1, 2, 4, 6, and 12 weeks
- **Multiple imputation strategies** — median fill, IterativeImputer with BayesianRidge, IterativeImputer with SVR
- **Outlier analysis** — full pipeline run both with and without IQR-based outlier removal
- **Automatic feature engineering** — lag features, rolling statistics (mean, std, min, max), and percent-change features
- **Hyperparameter optimization** — Optuna studies with 15 trials per model configuration
- **Full experiment tracking** — every run logged to MLflow with parameters, metrics, models, scalers, imputers, SHAP plots, and feature importance tables
- **Model interpretation** — SHAP summary/bar/scatter plots for best models

## Tech Stack

| Category | Tools |
|----------|-------|
| **Models** | LightGBM, CatBoost |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow |
| **Imputation** | IterativeImputer (BayesianRidge, SVR) |
| **Interpretability** | SHAP |
| **Time Series** | Darts |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Python 3.12, uv |

## Pipeline Overview

```
Raw Data (Rossmann)
  │
  ├─ Train / Validation split (85/15 by date)
  │
  ├─ Imputation
  │   ├─ Group median fill
  │   ├─ IterativeImputer + BayesianRidge
  │   └─ IterativeImputer + SVR
  │
  ├─ Feature Engineering
  │   ├─ Lag features (SalesLag1..SalesLagN)
  │   ├─ Rolling statistics (mean, std, min, max)
  │   └─ Percent-change & difference features
  │
  ├─ Aggregation (weekly / bi-weekly / monthly / quarterly)
  │
  ├─ Optuna Hyperparameter Search (×15 trials)
  │
  ├─ Model Training (LightGBM & CatBoost)
  │
  └─ MLflow Logging
      ├─ Parameters & metrics
      ├─ Serialized models, scalers, imputers
      ├─ Feature importance tables
      └─ SHAP plots
```

## Experiments Conducted

All 10 model configurations are trained **with** and **without** outlier removal (IQR-based per store):

| Aggregation | Forecast Steps | LightGBM | CatBoost |
|-------------|---------------|----------|----------|
| 1 week      | 12 steps      | ✅       | ✅       |
| 2 weeks     | 6 steps       | ✅       | ✅       |
| 4 weeks     | 3 steps       | ✅       | ✅       |
| 6 weeks     | 2 steps       | ✅       | ✅       |
| 12 weeks    | 1 step        | ✅       | ✅       |

## Results

Best model comparison by **WAPE** (Weighted Absolute Percentage Error):

Results and SHAP interpretation plots are logged to MLflow and available as artifacts in the `comparison/` and `shap/` directories.

## Installation

```bash
# Install uv (if not installed)
pip install uv

# Sync environment
uv sync
```

## Usage

Run the full pipeline in the Jupyter notebook:

```bash
jupyter notebook project.ipynb
```

Or run MLflow UI to explore experiment results:

```bash
mlflow ui
```

---

## RU

# 🏪 Прогнозирование продаж Rossmann

Пайплайн многогоризонтного прогнозирования продаж для датасета [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) с использованием **LightGBM** и **CatBoost**, гиперпараметрической оптимизацией через **Optuna**, трекингом экспериментов в **MLflow** и интерпретацией моделей через **SHAP**.

### Ключевые возможности

- **Многогоризонтное прогнозирование** — модели обучены на 5 уровнях агрегации: 1, 2, 4, 6 и 12 недель
- **Различные стратегии импутации** — заполнение медианой, IterativeImputer с BayesianRidge, IterativeImputer с SVR
- **Анализ выбросов** — полный прогон пайплайна с удалением выбросов (IQR) и без
- **Автоматический инжиниринг признаков** — лаговые признаки, скользящие статистики (среднее, std, мин, макс), процентные изменения
- **Оптимизация гиперпараметров** — Optuna studies по 15 trials на конфигурацию модели
- **Полный трекинг экспериментов** — каждый запуск логируется в MLflow с параметрами, метриками, моделями, масштабаторами, импутаторами, SHAP-графиками и таблицами важности признаков
- **Интерпретация моделей** — SHAP summary/bar/scatter графики для лучших моделей

### Технологический стек

| Категория | Инструменты |
|-----------|-------------|
| **Модели** | LightGBM, CatBoost |
| **Оптимизация** | Optuna |
| **Трекинг** | MLflow |
| **Импутация** | IterativeImputer (BayesianRidge, SVR) |
| **Интерпретация** | SHAP |
| **Time Series** | Darts |
| **Обработка данных** | Pandas, NumPy, Scikit-learn |
| **Визуализация** | Matplotlib, Seaborn |
| **Окружение** | Python 3.12, uv |

### Схема пайплайна

```
Сырые данные (Rossmann)
  │
  ├─ Разделение train / validation (85/15 по дате)
  │
  ├─ Импутация
  │   ├─ Заполнение медианой по группе
  │   ├─ IterativeImputer + BayesianRidge
  │   └─ IterativeImputer + SVR
  │
  ├─ Инжиниринг признаков
  │   ├─ Лаговые признаки (SalesLag1..SalesLagN)
  │   ├─ Скользящие статистики (среднее, std, мин, макс)
  │   └─ Процентные изменения и разности
  │
  ├─ Агрегация (неделя / 2 недели / месяц / квартал)
  │
  ├─ Поиск гиперпараметров через Optuna (×15 trials)
  │
  ├─ Обучение модели (LightGBM и CatBoost)
  │
  └─ Логирование в MLflow
      ├─ Параметры и метрики
      ├─ Сериализованные модели, масштабаторы, импутаторы
      ├─ Таблицы важности признаков
      └─ SHAP-графики
```

### Проведенные эксперименты

Все 10 конфигураций моделей обучены **с удалением выбросов** и **без него** (IQR по каждому магазину):

| Агрегация | Шагов прогноза | LightGBM | CatBoost |
|-----------|---------------|----------|----------|
| 1 неделя  | 12 шагов      | ✅       | ✅       |
| 2 недели  | 6 шагов       | ✅       | ✅       |
| 4 недели  | 3 шага        | ✅       | ✅       |
| 6 недель  | 2 шага        | ✅       | ✅       |
| 12 недель | 1 шаг         | ✅       | ✅       |

### Результаты

Сравнение лучших моделей по метрике **WAPE** (Weighted Absolute Percentage Error):

Результаты и SHAP-графики сохраняются в MLflow и доступны как артефакты в директориях `comparison/` и `shap/`.

### Установка

```bash
pip install uv
uv sync
```

### Запуск

```bash
jupyter notebook project.ipynb
# или
mlflow ui
```
