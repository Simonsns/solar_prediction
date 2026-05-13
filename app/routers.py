from fastapi import APIRouter, Depends, HTTPException
from typing import Any
from app.dependencies import get_supabase_service
from app.schemas import LatestSolarPredictionResponse, SolarPredictionItem
from src.services.supabase_api import SupabaseAPIService

router = APIRouter(prefix="/solar", tags=["Solar Predictions"])

@router.get("/predictions/latest", response_model=LatestSolarPredictionResponse)
def get_latest_solar_predictions(
    supabase_svc: SupabaseAPIService = Depends(get_supabase_service)
) -> Any:
    """
    Return the latest predictions.
    """
    data = supabase_svc.get_latest_predictions(table_name="predictions")
    
    if not data:
        raise HTTPException(status_code=404, detail="No data available")

    predicted_at = data[0]["predicted_at"]
    
    predictions_list = [
        SolarPredictionItem(
            forecast_horizon=row["forecast_horizon"],
            predicted_value=row["predicted_value"]
        )
        for row in data
    ]

    return LatestSolarPredictionResponse(
        predicted_at=predicted_at,
        predictions=predictions_list
    )