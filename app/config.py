from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        )
    
    # Credentials
    supabase_api_url: SecretStr
    supabase_api_key: SecretStr
    
    # Others
    project_name: str = "Solar forecasting predictions Serving API"
    version: str = "1.0.0"
    
settings = APISettings()  # type: ignore