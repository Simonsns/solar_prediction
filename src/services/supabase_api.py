import logging
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import APISettings
logger = logging.getLogger(__name__)

class SupabaseAPIService:
    """Supabase service for API"""
    def __init__(self, settings: APISettings):

        self._client: Client = create_client(
            settings.supabase_api_url.get_secret_value(), 
            settings.supabase_api_key.get_secret_value()
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