#%%
from datetime import datetime, timedelta
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any

class SolarSettings(BaseSettings):
    """Settings for inference and training pipeline"""

    ### ENV
    model_config = SettingsConfigDict(env_file='.env', 
                                      extra="ignore")

    ### PARAMÈTRES GÉNÉRAUX D'EXÉCUTION
    TIME_AGREGATION: str = "1h"
    N_HOURS_TO_FETCH: int = Field(default=99, gt=0)
    LEN_PREV: int = 24
    CENTRAL_SCENARIO: int = 13 # Scénario barycentrique de toute la capacité solaire régionale
    REGION_CODE: int = Field(default=76, gt=0) # Occitanie

    # FEATURE ENGINEERING
    LAG_LIST: List[int] = Field(default=[1, 6, 24, 48], min_length=1)
    WINDOW_LIST: List[int] = Field(default = [6, 24, 48], min_length=1)
    
    # PARAMETRES API
    API_WEATHER_VARIABLES: List[str] = Field(min_length=1)
    LAGGED_FEATURE_LIST: List[str] = Field(min_length=1)
    TIMEFRAME_DICT: Dict[str, int] = Field(min_length=1)

    # DYNAMIC VARIABLES
    @computed_field
    @property
    def DATE_START_CALC(self) -> datetime:
        """Departure calcul at instanciation time"""
        return datetime.now() - timedelta(hours=self.N_HOURS_TO_FETCH)

    @computed_field
    @property
    def DATE_START_STR(self) -> str:
        """Time formatting of API Open-Meteo response"""
        return self.DATE_START_CALC.strftime("%Y-%m-%d %H:%M:%S")

    @computed_field
    @property
    def RTE_DEFAULT_PARAMS(self) -> Dict[str, Any]:
        """Generates dynamically default rte parameters"""
        return {
            "select": "code_insee_region, date, heure, date_heure, solaire",
            "order_by": "date_heure",
            "where": f"code_insee_region='{self.REGION_CODE}' AND date_heure >= '{self.DATE_START_STR}'"
        }
