import os
import io
import requests
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

API_URL = "http://93.81.248.105:18080/api/predict-csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../ml/scripts
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
CSV_PATH = os.path.normpath(CSV_PATH)  # clean up ".."
BATCH_SIZE = 10_000


def call_api_on_df(df_batch: pd.DataFrame) -> List[Dict[str, Any]]:
    # Serialize batch to CSV in-memory
    buf = io.StringIO()
    df_batch.to_csv(buf, index=False)
    buf.seek(0)

    files = {
        "file": ("batch.csv", buf.getvalue().encode("utf-8"), "text/csv"),
    }
    resp = requests.post(API_URL, files=files, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # unpack the wrapper: {"predictions": [...], "rows": N}
    return data["predictions"]


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.head(5000000)
    data_test, data_val = train_test_split(df, test_size=0.1, stratify=df["Response"])
    assert "id" in df.columns and "Response" in df.columns

    all_preds: List[Dict[str, Any]] = []

    for start in range(0, len(data_val), BATCH_SIZE):
        end = start + BATCH_SIZE
        df_batch = data_val.iloc[start:end].copy()
        print(f"Sending rows {start}..{end-1} ({len(df_batch)})")

        batch_preds = call_api_on_df(df_batch)
        all_preds.extend(batch_preds)

    df_pred = pd.DataFrame(all_preds)
    # merge by id to align ground truth and predictions
    merged = data_val.merge(
        df_pred[["id", "Response"]], on="id", suffixes=("_true", "_pred")
    )

    y_true = merged["Response_true"].astype(int)
    y_pred = merged["Response_pred"].astype(int)

    print("Classification report (API batched vs ground truth):")
    print(classification_report(y_true, y_pred))
    print(f"F1 score: {f1_score(y_true, y_pred):.3f}")


if __name__ == "__main__":
    main()
