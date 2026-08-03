import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skops.io import get_untrusted_types, load as skops_load

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


REQUIRED_COLUMNS = [
    "Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
    "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
]

MONTHS_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def load_artifacts(model_dir="best_model"):
    untrusted = get_untrusted_types(file=f"{model_dir}/model.skops")
    model = skops_load(f"{model_dir}/model.skops", trusted=untrusted)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(f"{model_dir}/best_config.json") as f:
        config = json.load(f)

    return model, scaler, imputer, config


def validate_columns(df: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return {"missing": missing, "extra": extra}


def validate_data(df: pd.DataFrame) -> list[str]:
    errors = []

    required_no_nan = {
        "Store", "Date", "Sales", "Open", "Promo",
        "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "Promo2",
    }

    for col in required_no_nan:
        if col not in df.columns:
            continue
        nan_rows = df.index[df[col].isna()].tolist()
        if nan_rows:
            nums = ", ".join(str(r + 2) for r in nan_rows[:5])
            suffix = f" и ещё {len(nan_rows) - 5}" if len(nan_rows) > 5 else ""
            errors.append(f"Колонка '{col}': пустые значения в строках {nums}{suffix}")

    if "Store" in df.columns:
        bad = df[~df["Store"].apply(lambda x: isinstance(x, (int, float)) and x == int(x) and x > 0)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"Store: должен быть целым числом > 0 (строки {nums})")

    if "DayOfWeek" in df.columns:
        bad = df[~df["DayOfWeek"].apply(lambda x: isinstance(x, (int, float)) and x == int(x) and 1 <= x <= 7)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"DayOfWeek: должно быть целым числом 1–7 (строки {nums})")

    if "Customers" in df.columns:
        non_numeric = df[~df["Customers"].apply(lambda x: isinstance(x, (int, float)))]
        negative = df[df["Customers"] < 0] if df["Customers"].dtype in ("int64", "float64") else pd.DataFrame()
        if len(non_numeric):
            nums = ", ".join(str(i + 2) for i in non_numeric.index[:5])
            errors.append(f"Customers: должно быть числом (строки {nums})")
        if len(negative):
            nums = ", ".join(str(i + 2) for i in negative.index[:5])
            errors.append(f"Customers: не может быть отрицательным (строки {nums})")

    if "Sales" in df.columns:
        non_numeric = df[~df["Sales"].apply(lambda x: isinstance(x, (int, float)))]
        negative = df[df["Sales"] < 0] if df["Sales"].dtype in ("int64", "float64") else pd.DataFrame()
        if len(non_numeric):
            nums = ", ".join(str(i + 2) for i in non_numeric.index[:5])
            errors.append(f"Sales: должно быть числом (строки {nums})")
        if len(negative):
            nums = ", ".join(str(i + 2) for i in negative.index[:5])
            errors.append(f"Sales: не может быть отрицательным (строки {nums})")

    for col in ("Open", "Promo", "SchoolHoliday", "Promo2"):
        if col in df.columns:
            bad = df[~df[col].isin([0, 1])]
            if len(bad):
                nums = ", ".join(str(i + 2) for i in bad.index[:5])
                errors.append(f"{col}: должно быть 0 или 1 (строки {nums})")

    if "StateHoliday" in df.columns:
        valid = {"0", "a", "b", "c"}
        bad = df[~df["StateHoliday"].astype(str).isin(valid)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"StateHoliday: допустимые значения 0, a, b, c (строки {nums})")

    if "StoreType" in df.columns:
        valid = {"a", "b", "c", "d"}
        bad = df[~df["StoreType"].astype(str).isin(valid)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"StoreType: допустимые значения a, b, c, d (строки {nums})")

    if "Assortment" in df.columns:
        valid = {"a", "b", "c"}
        bad = df[~df["Assortment"].astype(str).isin(valid)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"Assortment: допустимые значения a, b, c (строки {nums})")

    if "CompetitionDistance" in df.columns:
        col = df["CompetitionDistance"].dropna()
        bad = col[col < 0]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"CompetitionDistance: не может быть отрицательным (строки {nums})")

    if "CompetitionOpenSinceMonth" in df.columns:
        col = df["CompetitionOpenSinceMonth"].dropna()
        bad = col[(col < 1) | (col > 12)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"CompetitionOpenSinceMonth: должно быть 1–12 (строки {nums})")

    if "Promo2SinceWeek" in df.columns:
        col = df["Promo2SinceWeek"].dropna()
        bad = col[(col < 0) | (col > 53)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"Promo2SinceWeek: должно быть 0–53 (строки {nums})")

    if "Promo2SinceYear" in df.columns:
        col = df["Promo2SinceYear"].dropna()
        bad = col[(col < 0) | ((col > 0) & (col < 1900))]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"Promo2SinceYear: должно быть 0 или >= 1900 (строки {nums})")

    if "PromoInterval" in df.columns:
        valid_months = set(MONTHS_MAP.keys())
        def check_interval(val):
            if pd.isna(val):
                return True
            parts = [p.strip() for p in str(val).split(",")]
            return all(p in valid_months for p in parts)
        bad = df[~df["PromoInterval"].apply(check_interval)]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"PromoInterval: формат 'Jan,Apr,Jul,Oct' (строки {nums})")

    if "Date" in df.columns:
        bad = df[df["Date"].isna()]
        if len(bad):
            nums = ", ".join(str(i + 2) for i in bad.index[:5])
            errors.append(f"Date: некорректная дата (строки {nums})")

    return errors


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.day_of_year

    comp_dates = pd.to_datetime({
        "year": df["CompetitionOpenSinceYear"],
        "month": df["CompetitionOpenSinceMonth"],
        "day": 1,
    })
    diff = (df["Date"].dt.to_period("M") - comp_dates.dt.to_period("M"))
    df["CompetitionMonths"] = diff.apply(
        lambda x: np.nan if pd.isna(x) else x.n
    )

    df["PromoInterval"] = df["PromoInterval"].replace({"NaN": np.nan, "": np.nan})
    df["PromoInterval"] = df["PromoInterval"].str.split(",").apply(
        lambda row: [MONTHS_MAP[x] for x in row] if isinstance(row, list) else [0]
    )
    df["Promo2Exists"] = df.apply(
        lambda row: row["Month"] in row["PromoInterval"], axis=1
    )

    mask = df["Promo2"] == 1
    years = df.loc[mask, "Promo2SinceYear"].astype(int).astype(str)
    weeks = df.loc[mask, "Promo2SinceWeek"].astype(int).astype(str)
    df.loc[mask, "Promo2StartDate"] = pd.to_datetime(
        years + "-" + weeks + "-1", format="%Y-%W-%w", errors="coerce"
    )

    df["IsPromo2Month"] = (
        (df["Promo2"].astype(bool))
        & (df["Date"] >= df["Promo2StartDate"])
        & (df["Promo2Exists"])
    )
    df = df.drop(columns=["PromoInterval", "Promo2Exists", "Promo2StartDate"])

    df["PromoMean7"] = df.groupby("Store")["Promo"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    df["PromoMean28"] = df.groupby("Store")["Promo"].transform(
        lambda s: s.shift(1).rolling(28).mean()
    )
    df["SchoolHolidaySum7"] = df.groupby("Store")["SchoolHoliday"].transform(
        lambda s: s.shift(1).rolling(7).sum()
    )
    df["SchoolHolidaySum28"] = df.groupby("Store")["SchoolHoliday"].transform(
        lambda s: s.shift(1).rolling(28).sum()
    )

    df["StoreType"] = df["StoreType"].map({"a": 1, "b": 2, "c": 3, "d": 4})
    df["Assortment"] = df["Assortment"].map({"a": 1, "b": 2, "c": 3})
    df["StateHoliday"] = df["StateHoliday"].map({"0": 0, "a": 1, "b": 2, "c": 3})

    if "Customers" in df.columns:
        df = df.drop(columns=["Customers"])
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


def preprocess(df: pd.DataFrame, scaler, imputer, config: dict) -> pd.DataFrame:
    df = df.copy()
    bool_features = config["bool_features"]
    cat_features = config["cat_features"]
    num_features = config["num_features"]
    target_columns = config["target_columns"]

    X_num = df[num_features]
    X_scaled = scaler.transform(X_num)
    X_imputed = imputer.transform(X_scaled)
    X_restored = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=num_features, index=df.index)

    for col in ["CompetitionMonths", "SchoolHolidaySum7", "SchoolHolidaySum28"]:
        X_restored[col] = X_restored[col].astype(int)

    result = df[cat_features + bool_features + ["Sales", "Date"]]
    result = result.join(X_restored)

    return result[target_columns + ["Sales", "Date"]]


def get_period_config(n_weeks: int) -> dict:
    PERIOD_CONFIG = {
        1:  {"lags": [1, 2, 3, 4], "windows": [4, 12], "pct_change": [4, 12]},
        2:  {"lags": [1, 2, 3],    "windows": [2, 6],  "pct_change": [2, 6]},
        4:  {"lags": [1, 2, 3],    "windows": [3, 6],  "pct_change": [1, 3]},
        6:  {"lags": [1, 2],       "windows": [2, 4],  "pct_change": [1, 2]},
        12: {"lags": [1, 2],       "windows": [2, 3],  "pct_change": [1, 2]},
    }

    return PERIOD_CONFIG[n_weeks]


def aggregate_bucket(full_df: pd.DataFrame, n_weeks: int, agg_dict: dict) -> pd.DataFrame:
    tmp = full_df.copy()
    global_min_week = tmp["Week"].min()
    tmp["WeekNum"] = (tmp["Week"] - global_min_week).apply(lambda x: np.nan if pd.isna(x) else x.n)
    tmp["Bucket"] = tmp["WeekNum"] // n_weeks
    tmp = tmp.drop(columns=["WeekNum"])

    bucket_df = tmp.groupby(["Store", "Bucket"]).agg(agg_dict)
    bucket_df["n_days"] = tmp.groupby(["Store", "Bucket"])["Date"].count()
    bucket_df = bucket_df[bucket_df["n_days"] == n_weeks * 7]
    bucket_df = bucket_df.drop(columns=["n_days"])

    return bucket_df.reset_index()


def get_target_feature_cols(n_weeks: int) -> list[str]:
    cfg = get_period_config(n_weeks)
    cols = [f"SalesLag{lag}" for lag in cfg["lags"]]
    for w in cfg["windows"]:
        cols += [f"SalesMean{w}", f"SalesStd{w}", f"SalesMin{w}", f"SalesMax{w}"]
    for periods in cfg["pct_change"]:
        cols += [f"SalesPctChange{periods * n_weeks}w", f"SalesDiff{periods * n_weeks}w"]
        
    return cols


def add_target_features(data: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    data = data.sort_values(['Store', 'Date'])
    cfg = get_period_config(n_weeks)

    for lag in cfg['lags']:
        data[f'SalesLag{lag}'] = data.groupby('Store')['Sales'].shift(lag)

    for window in cfg['windows']:
        data[f'SalesMean{window}'] = data.groupby('Store')['Sales'].transform(lambda s: s.shift(1).rolling(window).mean())
        data[f'SalesStd{window}']  = data.groupby('Store')['Sales'].transform(lambda s: s.shift(1).rolling(window).std())
        data[f'SalesMin{window}']  = data.groupby('Store')['Sales'].transform(lambda s: s.shift(1).rolling(window).min())
        data[f'SalesMax{window}']  = data.groupby('Store')['Sales'].transform(lambda s: s.shift(1).rolling(window).max())

    for periods in cfg['pct_change']:
        weeks_back = periods * n_weeks
        data[f'SalesPctChange{weeks_back}w'] = data.groupby('Store')['Sales'].transform(
            lambda s: s.shift(1).pct_change(periods).replace(np.inf, np.nan)
        )
        data[f'SalesDiff{weeks_back}w'] = data.groupby('Store')['Sales'].transform(
            lambda s: s.shift(1).diff(periods)
        )

    return data


def predict(
    model: LGBMRegressor,
    data: pd.DataFrame,
    n_steps: int,
    n_weeks: int,
) -> pd.DataFrame:
    feature_cols = get_target_feature_cols(n_weeks)
    drop_cols = ['Sales', 'Date', 'Bucket']
    
    last_known_bucket = data.groupby('Store')['Bucket'].max().to_dict()
    data = add_target_features(data, n_weeks)

    for _ in range(n_steps):
        current_idx = data.groupby('Store')['Bucket'].idxmax()
        X_step = data.loc[current_idx].copy()
        X_step = X_step.dropna(subset=feature_cols)
        
        X_pred = X_step.drop(columns=drop_cols, errors='ignore')
        y_pred = model.predict(X_pred)
        
        new_rows = X_step.copy()
        new_rows['Sales'] = y_pred
        new_rows['Bucket'] = new_rows['Bucket'] + 1
        last_dates = data.groupby('Store')['Date'].max()
        new_rows['Date'] = new_rows['Store'].map(last_dates) + pd.Timedelta(days=n_weeks * 7)
        
        data = pd.concat([data, new_rows], ignore_index=True)
        data = add_target_features(data, n_weeks)
    
    pred_mask = data.apply(lambda row: row['Bucket'] > last_known_bucket.get(row['Store'], -1), axis=1)
    pred_final = (
        data[pred_mask]
        .set_index(['Store', 'Bucket'])[['Sales', 'Date']]
        .reset_index()
    )
    
    return pred_final


def load_history(user_min_date: pd.Timestamp, n_weeks: int, store_ids: list) -> pd.DataFrame:
    cfg = get_period_config(n_weeks)
    min_buckets = max(max(cfg["windows"]), max(cfg["pct_change"])) + 1
    min_days = min_buckets * n_weeks * 7
    cutoff = pd.Timestamp(user_min_date) - pd.Timedelta(days=min_days)

    url = os.getenv("DATABASE_URL")
    engine = create_engine(url)
    placeholders = ", ".join(["%s"] * len(store_ids))
    query = f"SELECT * FROM history WHERE Date >= %s AND Store IN ({placeholders})"

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=(cutoff, *store_ids), parse_dates=["Date"])

    df.columns = REQUIRED_COLUMNS
    return df


def run_pipeline(
    df_input: pd.DataFrame,
    model: LGBMRegressor,
    scaler,
    imputer,
    config: dict,
) -> pd.DataFrame:
    df_input["Date"] = pd.to_datetime(df_input["Date"])

    stores = df_input["Store"].unique().tolist()
    history = load_history(df_input["Date"].min(), config["n_weeks"], stores)
    df = pd.concat([history, df_input], ignore_index=True)
    df = df.drop_duplicates(subset=["Store", "Date"], keep="last")

    df = feature_engineering(df)
    df = preprocess(df, scaler, imputer, config)
    df["Week"] = df["Date"].dt.to_period("W-MON")

    bucketed = aggregate_bucket(df, n_weeks=config["n_weeks"], agg_dict=config["agg_dict"])
    bucketed = bucketed[bucketed["Store"].isin(stores)]

    bucketed = bucketed.astype({
        "Store": "category",
        "StoreType": "category",
        "Assortment": "category",
        "StateHoliday": "category",
        "DayOfWeek": "category",
        "Promo": "bool",
        "Promo2": "bool",
        "SchoolHoliday": "bool",
        "IsPromo2Month": "bool",
        "Promo2SinceWeek": "int",
        "Promo2SinceYear": "int",
        "Sales": "float",
    })

    return predict(model, bucketed, config["n_steps"], config["n_weeks"])
