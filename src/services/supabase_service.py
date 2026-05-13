# SupabaseService for initialization and functions
# author : simon.senegas
# date : 12/11/2025

import io
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.config import SolarSettings

logger = logging.getLogger(__name__)

class SupabaseService:

    def __init__(self, settings: SolarSettings):
        self._client: Client = create_client(
            supabase_url=settings.supabase_url.get_secret_value(),
            supabase_key=settings.supabase_key.get_secret_value(),
        ) 

    # 1 - Table

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def extract_coordinates(
            self,
            table_name: str
            ):
        """
        Fetch solar capacity barycentre coordinates and return a GeoDataFrame with data.
        """
        response = self._client.table(table_name).select("id", "geometry").execute()
        if not response.data:
            raise ValueError(f"No rows in {table_name}")
        
        df = pd.DataFrame(response.data)
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        logger.debug("Fetched %d coordinates from '%s'", len(gdf), table_name)

        return gdf

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def extract_quantile(
            self,
            table_name: str
            ):
        """
        Extract the last computed quantile.
        """
        response = ( 
            self._client
            .table(table_name)
            .select("p99_value")
            .order("computed_at", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:
            logger.error(f'No data found in the table {table_name}')
            raise ValueError("No data returned")
        
        quantile_value = response.data[0]
        raw = quantile_value["p99_value"] # type: ignore

        if not isinstance(raw, (int, float)) or not np.isfinite(float(raw)):
            raise ValueError(f'Invalid value for p99 regional quantile production : {raw!r}')
        
        logger.info(f"p99_value={float(raw)}")
        return float(raw)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def upsert_predictions(
            self,
            table_name: str,
            predictions: pd.DataFrame
    ) -> None:
       
        # Init
        df_to_upload = predictions.copy()
        for col in df_to_upload.select_dtypes(include=['datetimetz']).columns:
            df_to_upload[col] = df_to_upload[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        records = df_to_upload.to_dict(orient="records")

        # Upsert
        response = (
            self._client.table(table_name)
            .upsert(records) #type: ignore
            .execute()
        )
        
        logger.info(
            "[SUCCESS] Upserted %d rows into '%s'", 
            len(records), table_name
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def get_latest_predictions(self, table_name: str):
        """
        Fetch the most recent 24h predictions based on the latest predicted_at timestamp.
        """
        # Latest predicted_at
        latest_time_response = (
            self._client.table(table_name)
            .select("predicted_at")
            .order("predicted_at", desc=True)
            .limit(1)
            .execute()
        )

        if not latest_time_response.data:
            logger.warning(f"Empty table {table_name}")
            return []

        latest_predicted_at = latest_time_response.data[0]["predicted_at"] #type: ignore
        logger.info(f"Latest prediction at : {latest_predicted_at}")

        # Predictions
        predictions_response = (
            self._client.table(table_name)
            .select("predicted_at, forecast_horizon, predicted_value")
            .eq("predicted_at", latest_predicted_at)
            .order("forecast_horizon", desc=False)
            .execute()
        )

        return predictions_response.data 

    # 2 - Storage 
    
    def upload_artifact(
            self,
            dataset: pd.DataFrame,
            bucket_name: str,
            file_path: str = "latest_dataset.parquet"
            ) -> None:
        """Upload the latest dataset into bucket artifact on Supabase"""
        
        # Buffer parquet
        buffer = io.BytesIO()
        dataset.to_parquet(buffer, index=True, engine='pyarrow')

        # Upload
        response = self._client.storage.from_(bucket_name).upload(
            path=file_path,
            file=buffer.getvalue(),
            file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
            )
        
        logger.info("[SUCCESS] Uploaded %s → %s/%s", dataset.shape, bucket_name, file_path)

    def download_artifact(
            self,
            bucket_name: str,
            file_path: str = "latest_dataset.parquet"
            ) -> pd.DataFrame:
        """Download the latest dataset from Supabase bucket and return a DataFrame"""
        
        # Download
        response = self._client.storage.from_(bucket_name).download(
            path=file_path
            )
        
        # Buffer parquet
        buffer = io.BytesIO(response)
        dataset = pd.read_parquet(buffer, engine="pyarrow") # type: ignore
        
        logger.info(f"[SUCCESS] Dataset downloaded : {bucket_name}/{file_path}")
        logger.info("Downloaded %s from %s/%s", dataset.shape, bucket_name, file_path)

        return dataset
    
    def __repr__(self) -> str:
        return f"SupabaseService(initialized=True)"