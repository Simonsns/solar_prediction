#%%
# Dependencies
import pandas as pd
import logging
from typing import Tuple
logger = logging.getLogger(__name__)

# Modules 
from src.services.supabase_service import SupabaseService
from src.utils.config import SolarSettings
from src.etl.data_processing import solar_preprocessing, feature_engine
from src.etl.data_collection import fetching_solar_data, fetching_weather_data

# Schemas
from src.etl import schemas

class SolarETLInferenceJob:
    """Solar orchestrator for ETL"""
    def __init__(self, config: SolarSettings):
        self.config = config
        self._supabase = SupabaseService(settings=config)
    
    def extract(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("[EXTRACT] Phase - Beginning...")
        
        # 1 - Coordinates extraction
        logger.info("[EXTRACT] coordinates from PostgreSQL database...")
        coordinates = self._supabase.extract_coordinates(
            table_name=self.config.coord_table.get_secret_value()
        )
        logger.info(f"[SUCCESS] - [EXTRACT] '{len(coordinates)}' coordinates succeeded")

        # 2 - Solar production timeseries 
        logger.info(f"[EXTRACT] solar production records")
        raw_solar_prod = fetching_solar_data.fetch_inference_solar_data(
            url=self.config.api_solar_key.get_secret_value(),
            n_records=self.config.n_hours_to_fetch*4, # fifteen minutes
            params=self.config.rte_inference_params
        )
        
        prod_data = solar_preprocessing.prepare_production_data(
            production_data=raw_solar_prod,
            code_region=self.config.region_code,
            time_agregation=self.config.time_agregation,
            data_type="inference"
        )
        #prod_data = schemas.SolarProductionSchema.validate(prod_data)
        logger.info(f"[SUCCESS] - [EXTRACT] {len(prod_data)} solar production records extracted")

        # 3 - Meteo (past and forecast)
        common_weather_params = {
            'production_data': prod_data,
            'variables': self.config.api_weather_variables,
            "coordinates": coordinates,
        }
        # Past meteo features
        logger.info("[EXTRACT] Historical meteo features")
        past_weather = fetching_weather_data.fetch_historical_weather(
            **common_weather_params,
            weather_url=self.config.api_weather_key.get_secret_value()
        )
        #past_weather = schemas.WeatherPastModel.validate(past_weather)
        logger.info("[SUCCESS] - [EXTRACT] Historical meteo features")

        # Forecast meteo features
        logger.info("[EXTRACT] Forecast meteo features")
        forecast_weather = fetching_weather_data.fetch_forecast_weather(
            **common_weather_params,
            len_prev=self.config.len_prev,
            forecast_weather_url=self.config.api_forecast_weather_key.get_secret_value()
        )
        logger.info("[SUCCESS] - [EXTRACT] Forecast meteo features")
        logger.info("[SUCCESS] - [EXTRACT] Phase succeeded")
        
        return prod_data, past_weather, forecast_weather
    
    def transform(
            self, 
            prod_data: pd.DataFrame, 
            past_weather: pd.DataFrame,
            forecast_weather: pd.DataFrame
            ) -> pd.DataFrame:
        
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
        
        # 2 - Feature engineering
        # - PAST
        past_inference_data = feature_engine.prepare_past_features(
            df=named_past_weather,
            target=prod_data,
            feature_list=self.config.past_feature_list,
            lag_list=self.config.past_lag_list,
            window_list=self.config.window_list
        )

        # - FORECAST
        forecast_inference_data = feature_engine.prepare_forecast_features(
            df=named_forecast_weather,
            feature_list=named_forecast_weather.columns,
            lag_list=self.config.forecast_lag_list,
            delta_list=self.config.forecast_delta_list
        )

        # Rearranging
        inference_data = pd.merge(
            past_inference_data,
            forecast_inference_data, 
            left_index=True, 
            right_index=True, 
            how="right"
            ).sort_index()
        
        # Inference only
        all_inference_data = ( inference_data
                          .drop(columns=named_forecast_weather.columns)
                          .loc[(inference_data.index >= named_past_weather.index[0]) &
                               (inference_data.index <= named_past_weather.index[-1])]
        )

        # 3 - Rearranging and finishing pipeline
        final_dataset = feature_engine.transform_pipeline(inference_data=all_inference_data,
                       timeframe_dict=self.config.timeframe_dict) 
        
        logger.info("[SUCCESS] - [TRANSFORM] Phase succeeded")
        logger.info(
            "[TRANSFORM] Dataset ready",
            extra={
                "rows": len(final_dataset),
                "columns": len(final_dataset.columns),
                "start": str(final_dataset.index.min()),
                "end": str(final_dataset.index.max()),
                "nulls": int(final_dataset.isna().sum().sum())
            }
        )
        
        return final_dataset
    
    def load(self, transformed_data: pd.DataFrame) -> None:
        logger.info("[LOAD] Phase - Beginning...")

        self._supabase.upload_artifact(
        dataset=transformed_data,
        bucket_name=self.config.inference_bucket_name.get_secret_value()
        )
    
        logger.info("[SUCCESS] - [LOAD] Phase succeeded")

    def run(self) -> pd.DataFrame:
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

        return final_dataset