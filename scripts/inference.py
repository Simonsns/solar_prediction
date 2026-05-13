"""
Pipeline ETL pour la prédiction de production solaire
author : Simon.sngs
date : 22/04/2026
"""

from src.utils.logger import setup_logging
setup_logging()
from src.utils.config import settings
from pipelines.inference_data_pipeline import SolarETLInferenceJob
from pipelines.inference_pipeline import InferenceJob
from src.services.supabase_service import SupabaseService

if __name__ == "__main__":
    
    # 0 - Init service
    supabase = SupabaseService(settings=settings)
    
    # 1 - ETL
    etl_instance = SolarETLInferenceJob(config=settings)
    latest_dataset = etl_instance.run()
    quantile_factor = supabase.extract_quantile(table_name="solar_p99")

    # 2 - Inference
    inference_instance = InferenceJob(config=settings)
    predictions = inference_instance.run(
        input_dataset=latest_dataset.drop(columns="solaire"),
        quantile_factor=quantile_factor
    )
    # 3 - Load
    supabase.upsert_predictions(
        table_name="predictions",
        predictions=predictions
    )