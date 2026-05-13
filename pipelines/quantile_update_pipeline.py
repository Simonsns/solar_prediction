# Modules 
from src.data_pipeline import schemas
from src.utils.config import SolarSettings
from src.data_pipeline.data_processing import solar_preprocessing
from src.data_pipeline.data_collection import fetching_solar_data
from datetime import datetime
from supabase import create_client, Client

# Miscellaneous
import pandas as pd
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class RollingQuantile:
    
    def __init__(
            self, 
            config: SolarSettings, 
            n_days: int = 90,
            quantile_value: float = 0.99
            ):
        
        # Init
        self.config = config
        self.n_days = n_days
        self.quantile_value = quantile_value
        self.supabase: Client = create_client(
            self.config.supabase_url.get_secret_value(), 
            self.config.supabase_key.get_secret_value())

        # Learnt value
        self.current_quantile_: Optional[float] = None

    def quantile_compute(self):
        """Compute the solar self.region_code production wanted quantile on last n_days"""
        
        logger.info(f"[QUANTILE] quantile{self.quantile_value} compute on last {self.n_days} days")
        raw_solar_prod = fetching_solar_data.fetch_inference_solar_data(
                    url=self.config.api_solar_key.get_secret_value(),
                    n_records=(self.n_days*24*4), # 90 days with 15 minutes interval
                    params={
                    "select": "code_insee_region, date, heure, date_heure, solaire",
                    "order_by": "date_heure",
                    "where": f"code_insee_region='{self.config.region_code}' AND date_heure >= '{str(datetime.now() - pd.Timedelta(days=self.n_days))}'"
                }
        )
        prod_data = solar_preprocessing.prepare_production_data(
            production_data=raw_solar_prod,
            code_region=self.config.region_code,
            time_agregation=self.config.time_agregation,
            data_type="inference"
        )
        
        self.current_quantile_ = prod_data["solaire"].quantile(self.quantile_value)
        logger.info(f"[SUCCESS] - [QUANTILE] quantile{self.quantile_value} computed on last {self.n_days} days")
        
        return self
    
    def upload_quantile(self):
        """Upload the quantile on SQL server"""

        if self.current_quantile_ is None:
            raise ValueError("No fitted quantile. Quantile must be calculated before upload")
        
        logger.info(f"[UPLOAD] quantile{self.quantile_value}: {self.current_quantile_} upload on SQL server")
        
        response = self.supabase.table("solar_p99").upsert({
            "region": self.config.region_code,
            "p99_value": self.current_quantile_,
            "computed_at": datetime.now().isoformat()
        }).execute()

        logger.info(f"[SUCCESS] quantile{self.quantile_value} uploaded on SQL server {response}")

    def run(self):
        """Rolling quantile compute and upload orchestrator"""
        self.quantile_compute()
        self.upload_quantile()