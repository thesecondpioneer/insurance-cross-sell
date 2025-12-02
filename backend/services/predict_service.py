import io
import pandas as pd
from typing import List, Dict, Any
from fastapi import UploadFile

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
    return [dict(row) for row in df.to_dict(orient="records")]


def predict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dummy prediction: randomly assign 0 or 1 to Response
    """
    import random

    predictions = []
    for row in data:
        new_row = row.copy()
        new_row["Response"] = random.choice([0, 1])
        predictions.append(new_row)
    return predictions
