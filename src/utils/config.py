from datetime import datetime, timedelta
from pydantic import Field, computed_field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any

class SolarSettings(BaseSettings):
    """Settings for inference and training pipeline"""

    ### ENV
    model_config = SettingsConfigDict(env_file='.env', 
                                      env_file_encoding='utf-8',
                                      extra="ignore")
    
    # Versioning
    project_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    
    # RTE and open-meteo API
    api_solar_key: SecretStr #Inference key
    api_hsolar_key: SecretStr #Training key
    api_weather_key: SecretStr
    api_forecast_weather_key: SecretStr
    api_hist_forecast_weather_key: SecretStr
    
    # Supabase credentials and bucket name
    supabase_url: SecretStr
    supabase_key: SecretStr
    coord_table: SecretStr
    inference_bucket_name: SecretStr
    training_bucket_name: SecretStr

    # MLflow credentials server
    mlflow_tracking_uri: SecretStr
    mlflow_tracking_username: SecretStr
    mlflow_tracking_password: SecretStr

    ### PARAMÈTRES GÉNÉRAUX D'EXÉCUTION
    time_agregation: str = "1h" # Agrégation des données (par défaut horaire)
    n_hours_to_fetch: int = Field(default=99, gt=0) # Nombre de records
    len_prev: int = 48 # Longueur des features prévisions (pour lags futurs)
    central_scenario: int = 13 # Scénario barycentrique de toute la capacité solaire régionale
    region_code: int = Field(default=76, gt=0) # Occitanie

    # FEATURE ENGINEERING
    # - PAST
    past_feature_list: List[str] = Field(min_length=1)
    past_lag_list: List[int] = Field(default=[1, 6, 24, 48], min_length=1)
    window_list: List[int] = Field(default = [6, 24, 48], min_length=1)
    timeframe_dict: Dict[str, int] = Field(min_length=1)

    # - FORECAST
    forecast_lag_list: List[int] = Field(default=[1, 3, 6, 9, 12, 16, 24], min_length=1)
    forecast_delta_list: List[int] = Field(default=[1, 6, 12, 24], min_length=1)
    
    # PARAMETRES API
    api_weather_variables: List[str] = Field(min_length=1)

    # DYNAMIC VARIABLES
    @computed_field
    @property
    def date_start_calc(self) -> datetime:
        """Departure calcul at instanciation time"""
        return datetime.now() - timedelta(hours=self.n_hours_to_fetch) # Remis en heures

    @computed_field
    @property
    def date_start_str(self) -> str:
        """Time formatting of API Open-Meteo response"""
        return self.date_start_calc.strftime("%Y-%m-%d %H:%M:%S")

    @computed_field
    @property
    def rte_inference_params(self) -> Dict[str, Any]:
        """Generates dynamically default rte parameters"""
        return {
            "select": "code_insee_region, date, heure, date_heure, solaire",
            "order_by": "date_heure",
            "where": f"code_insee_region='{self.region_code}' AND date_heure >= '{self.date_start_str}'"
        }

settings = SolarSettings() # type: ignore
