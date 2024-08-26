from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils.model_versioner import ModelVersioner
import pandas as pd
import numpy as np

app = FastAPI()
model_versioner = ModelVersioner()

class PredictionRequest(BaseModel):
    features: dict

class ModelInfo(BaseModel):
    model_name: str
    run_id: str = None

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    try:
        model = model_versioner.load_model(model_name)
        features = pd.DataFrame([request.features])
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info/{model_name}")
async def get_model_info(model_name: str):
    try:
        # This is a placeholder. You'll need to implement a method to get the latest run_id
        run_id = "latest_run_id"  
        return {"model_name": model_name, "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/{model_name}")
async def retrain_model(model_name: str):
    try:
        # This is a placeholder. You'll need to implement the retraining logic
        # It should fetch new data, retrain the model, and save it
        return {"message": f"Model {model_name} retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add more endpoints as needed