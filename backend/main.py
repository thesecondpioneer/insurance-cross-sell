from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from services.predict_service import parse_csv, predict

app = FastAPI(title="Insurance Prediction API")

# Allow CORS from frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-csv")
async def predict_csv(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        data: List[Dict[str, Any]] = parse_csv(file)
        predictions: List[Dict[str, Any]] = predict(data)
        return {"predictions": predictions, "rows": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
