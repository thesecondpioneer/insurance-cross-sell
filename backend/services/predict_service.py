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

with open(os.path.join(MODELS_DIR, "catboost_model_all_gpu.pkl"), "rb") as f:
    MODEL: CatBoostClassifier = pickle.load(f)


def parse_csv(file: UploadFile, max_rows: int = 10000) -> List[Dict[str, Any]]:
    """
    Parse CSV with safe type conversion per row
    """
    contents = file.file.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(contents))

    # Check required headers
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Limit rows
    df = df.head(max_rows)

    # Fill Response if missing
    if "Response" not in df.columns:
        df["Response"] = -1

    # Safe type conversion PER ROW
    def safe_convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
        row_copy = row.copy()

        # Numeric columns - try conversion, fail silently
        numeric_cols = [
            "id",
            "Age",
            "Driving_License",
            "Region_Code",
            "Previously_Insured",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
        ]

        for col in numeric_cols:
            if col in row_copy:
                try:
                    row_copy[col] = int(float(row_copy[col]))
                except (ValueError, TypeError):
                    row_copy[col] = None  # Will be caught by validation

        # String columns
        string_cols = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        for col in string_cols:
            if col in row_copy:
                row_copy[col] = str(row_copy[col]).strip()

        return row_copy

    # Apply safe conversion row by row
    rows = []
    for _, row in df.iterrows():
        rows.append(safe_convert_row(row.to_dict()))

    return rows


def validate_row(row: Dict[str, Any]) -> bool:
    # Check numeric columns exist and are valid numbers
    numeric_checks = {
        "Age": lambda x: isinstance(x, int) and 18 <= x <= 100,
        "Driving_License": lambda x: isinstance(x, int) and x in [0, 1],
        "Region_Code": lambda x: isinstance(x, int) and x >= 0,
        "Previously_Insured": lambda x: isinstance(x, int) and x in [0, 1],
        "Annual_Premium": lambda x: isinstance(x, int) and x >= 0,
        "Policy_Sales_Channel": lambda x: isinstance(x, int) and x >= 0,
        "Vintage": lambda x: isinstance(x, int) and x >= 0,
    }

    for col, check in numeric_checks.items():
        if col not in row or not check(row[col]):
            return False

    # String columns
    if (
        row.get("Gender") not in ["Male", "Female"]
        or row.get("Vehicle_Age") not in ["< 1 Year", "1-2 Year", "> 2 Years"]
        or row.get("Vehicle_Damage") not in ["Yes", "No"]
    ):
        return False

    return True


def predict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []

    # Keep original order
    results = []
    for row in data:
        if validate_row(row):
            # Make a copy for prediction
            row_copy = row.copy()
            results.append(row_copy)
        else:
            # Invalid row, mark as error
            row_copy = row.copy()
            row_copy["Response"] = "Error"
            results.append(row_copy)

    # Predict only for valid rows
    valid_rows = [r for r in results if r["Response"] != "Error"]

    if valid_rows:
        df_raw = pd.DataFrame(valid_rows)
        df_proc = apply_preprocessor(df_raw, PREPROCESSOR_PARAMS, is_train=False)

        pool = Pool(
            df_proc,
            cat_features=[c for c in CATEGORICAL_FEATURES if c in df_proc.columns],
        )

        proba = MODEL.predict_proba(pool)[:, 1]
        preds = (proba >= OPTIMAL_THRESHOLD).astype(int)

        for row, y_hat in zip(valid_rows, preds):
            row["Response"] = int(y_hat)

    return results
