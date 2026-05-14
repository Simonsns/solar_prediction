#%%
# Miscellaneous
import pandas as pd
import logging
import os

# Modules 
from src.services.supabase_service import SupabaseService
from src.utils.config import SolarSettings, settings
from src.etl.data_processing import solar_preprocessing, feature_engine
from src.etl.data_collection import fetching_solar_data, fetching_weather_data
from src.etl import schemas

logger = logging.getLogger(__name__)

class SolarETLTrainingJob:
    """Solar orchestrator for ETL"""
    def __init__(self, config: SolarSettings):
        self.config = config
        self._supabase = SupabaseService(settings=config)
    
    def extract(self): #-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("[EXTRACT] Phase - Beginning...")
        
        # 1 - Coordinates extraction
        logger.info("[EXTRACT] coordinates from PostgreSQL database...")
        coordinates = self._supabase.extract_coordinates(
            table_name=self.config.coord_table.get_secret_value()
        )
        logger.info(f"[SUCCESS] - [EXTRACT] '{len(coordinates)}' coordinates succeeded")

        # 2 - Solar production timeseries 
        # 2.1 - Fetching
        logger.info(f"[EXTRACT] solar production records")
        raw_solar_prod = fetching_solar_data.fetch_solar_data(
            url=self.config.api_hsolar_key.get_secret_value())
        
        prod_data = solar_preprocessing.prepare_production_data(
            production_data=raw_solar_prod,
            code_region=self.config.region_code,
            time_agregation=self.config.time_agregation,
            data_type="TRAINING",
            start_training_date="2021-01-01",
            end_training_date="2026-01-01"
        )

        logger.info(f"[SUCCESS] - [EXTRACT] {len(prod_data)} solar production records extracted")

        # 3 - Meteo (past and forecast)
        common_weather_params = {
            'variables': self.config.api_weather_variables,
            "coordinates": coordinates,
        }
        
        # Past meteo features
        logger.info("[EXTRACT] Historical meteo features")
        past_weather = fetching_weather_data.fetch_historical_weather(
            **common_weather_params,
            production_data=prod_data,
            weather_url=self.config.api_weather_key.get_secret_value()
        )
        logger.info("[SUCCESS] - [EXTRACT] Historical meteo features")

        # Forecast meteo features
        logger.info("[EXTRACT] Forecast meteo features")
        forecast_weather = fetching_weather_data.fetch_historical_forecast_weather(
            **common_weather_params,
            hist_forecast_start= prod_data.index.min(),
            hist_forecast_end=prod_data.index.max(),
            forecast_weather_url=self.config.api_hist_forecast_weather_key.get_secret_value()
        )
        if not prod_data.index.equals(past_weather.index):
            logger.warning("Mismatch between production and past weather index")

        logger.info("[SUCCESS] - [EXTRACT] Forecast meteo features")
        logger.info("[SUCCESS] - [EXTRACT] Phase succeeded")
        
        return prod_data, past_weather, forecast_weather
    
    def transform(
            self, 
            prod_data: pd.DataFrame, 
            past_weather: pd.DataFrame,
            forecast_weather: pd.DataFrame
            ): #-> pd.DataFrame:
        
        logger.info("[TRANSFORM] Phase - Beginning...")
        
        # 1 - Columns rename
        named_past_weather = feature_engine.col_scenario_rename(
            past_weather, 
            self.config.central_scenario
            )
        named_forecast_weather = feature_engine.col_scenario_rename(
            forecast_weather, 
            self.config.central_scenario
            )
        
        # Rearranging
        inference_data = ( pd.merge(
            named_past_weather,
            named_forecast_weather, 
            left_index=True,
            right_index=True
            )
            .sort_index() 
        )
        
        # 2 - Feature engineering
        # - PAST
        inference_data_with_past_features = feature_engine.prepare_past_features(
            df=inference_data,
            target=prod_data,
            feature_list=self.config.past_feature_list,
            lag_list=self.config.past_lag_list,
            window_list=self.config.window_list
        )

        # - FORECAST
        all_inference_data = feature_engine.prepare_forecast_features(
            df=inference_data_with_past_features,
            feature_list=named_forecast_weather.columns,
            lag_list=self.config.forecast_lag_list,
            delta_list=self.config.forecast_delta_list
        )
        # Suppress forecast between 0 and t
        all_inference_data = all_inference_data.drop(columns=named_forecast_weather.columns)

        # 3 - Rearranging and finishing pipeline
        final_dataset = feature_engine.transform_pipeline(inference_data=all_inference_data,
                       timeframe_dict=self.config.timeframe_dict) 
        
        logger.info("[SUCCESS] - [TRANSFORM] Phase succeeded")
        
        return final_dataset
    
    def load(self, transformed_data: pd.DataFrame) -> None:
        logger.info("[LOAD] Phase - Beginning...")

        self._supabase.upload_artifact(
        dataset=transformed_data, 
        bucket_name=self.config.training_bucket_name.get_secret_value())
    
        logger.info("[SUCCESS] - [LOAD] Phase succeeded")

    def run(self) -> None:
        """Execution run"""

        # Extract
        try:
            logger.info("="*50)
            logger.info("Job loading - Solar ETL Inference")
            prod_data, past_weather, forecast_weather  = self.extract()
        except Exception as e:
            logger.exception(f"[FAIL] Extract phase failed {str(e)}", exc_info=True)
            raise

        # Transform
        try:
            final_dataset = self.transform(
                prod_data=prod_data,
                past_weather=past_weather,
                forecast_weather=forecast_weather
            )
        except Exception as e:
            logger.exception(f"[FAIL] Transform phase failed {str(e)}", exc_info=True)
            raise
        
        # Load
        try:
            self.load(transformed_data=final_dataset) 
            logger.info("Solar ETL Inference job succeeded")
            logger.info("="*50)
        except Exception as e:
            logger.exception(f"[FAIL] Load phase failed {str(e)}", exc_info=True)
            raise