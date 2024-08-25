from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.model_versioner import ModelVersioner
from src.utils.logger import app_logger

app = FastAPI()

model_versioner = ModelVersioner('models')

class PredictionRequest(BaseModel):
    model_name: str
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    model_version: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        model, model_info = model_versioner.load_latest_model(request.model_name)
        prediction = model.predict([request.features])[0]
        return PredictionResponse(prediction=prediction, model_version=model_info['version'])
    except Exception as e:
        app_logger.error(f"Prediction error: {str(e)}")
        raise

@app.get("/models")
async def list_models():
    return model_versioner.versions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)