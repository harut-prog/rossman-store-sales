import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skops.io import get_untrusted_types, load as skops_load


REQUIRED_COLUMNS = [
    "Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
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


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


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
    df["CompetitionMonths"] = diff.apply(lambda x: x.n if pd.notna(x) else np.nan)

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
        (df["Promo2"])
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
    tmp["WeekNum"] = (tmp["Week"] - global_min_week).apply(lambda x: x.n)
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


def load_history(user_min_date: pd.Timestamp) -> pd.DataFrame:    
    df = pd.read_csv('history.csv', parse_dates=["Date"], low_memory=False)
    cutoff = user_min_date - pd.Timedelta(days=365)
    df = df[df["Date"] >= cutoff].reset_index(drop=True)
    
    return df


def run_pipeline(
    df_input: pd.DataFrame,
    model: LGBMRegressor,
    scaler,
    imputer,
    config: dict,
) -> pd.DataFrame:
    history = load_history(df_input["Date"].min())
    df = pd.concat([history, df_input], ignore_index=True)

    df = feature_engineering(df)
    df = preprocess(df, scaler, imputer, config)
    df["Week"] = df["Date"].dt.to_period("W-MON")

    bucketed = aggregate_bucket(df, n_weeks=config["n_weeks"], agg_dict=config["agg_dict"])
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
