import io
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import UploadFile
from catboost import CatBoostClassifier, Pool

from services.preprocessor import apply_preprocessor

REQUIRED_COLUMNS = [
    "id",
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

OPTIONAL_COLUMNS = ["Response"]

CATEGORICAL_FEATURES = [
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Age_Group",
    "Premium_Group",
]

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb") as f:
    PREPROCESSOR_PARAMS = pickle.load(f)

df_thr = pd.read_csv(os.path.join(MODELS_DIR, "Tresholds.csv"))

OPTIMAL_THRESHOLD = float(
    df_thr.loc[df_thr["Model"] == "catboost_model.pkl", "Treshold"].iloc[0]
)

with open(os.path.join(MODELS_DIR, "catboost_model.pkl"), "rb") as f:
    MODEL: CatBoostClassifier = pickle.load(f)


def parse_csv(file: UploadFile, max_rows: int = 10000) -> List[Dict[str, Any]]:
    """
    Parse uploaded CSV file using pandas (or streaming if needed)
    and enforce required headers. Limit rows to max_rows.
    """
    contents = file.file.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(contents))

    # Check required headers
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Limit rows to avoid huge files crashing memory
    df = df.head(max_rows)

    # Fill Response if missing
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = -1

    # Ensure correct types
    df["id"] = df["id"].astype(int)
    df["Age"] = df["Age"].astype(int)
    df["Driving_License"] = df["Driving_License"].astype(int)
    df["Region_Code"] = df["Region_Code"].astype(int)
    df["Previously_Insured"] = df["Previously_Insured"].astype(int)
    df["Annual_Premium"] = df["Annual_Premium"].astype(int)
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype(int)
    df["Vintage"] = df["Vintage"].astype(int)
    df["Response"] = df["Response"].astype(int)

    # Convert to list of dicts
    return df.to_dict(orient="records")  # type: ignore


def predict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []

    df_raw = pd.DataFrame(data)
    df_proc = apply_preprocessor(df_raw, PREPROCESSOR_PARAMS, is_train=False)

    pool = Pool(
        df_proc, cat_features=[c for c in CATEGORICAL_FEATURES if c in df_proc.columns]
    )

    proba = MODEL.predict_proba(pool)[:, 1]
    preds = (proba >= OPTIMAL_THRESHOLD).astype(int)

    out: List[Dict[str, Any]] = []
    for row, y_hat in zip(data, preds):
        new_row = row.copy()
        new_row["Response"] = int(y_hat)
        out.append(new_row)
    return out
