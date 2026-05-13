from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class SolarPredictionItem(BaseModel):
    forecast_horizon: datetime = Field(..., description="Targeted hour prediction")
    predicted_value: float = Field(..., description="Targeted production value")

class LatestSolarPredictionResponse(BaseModel):
    predicted_at: datetime = Field(..., description="Hour generation of predictions")
    predictions: List[SolarPredictionItem]