import mlflow
import pandas as pd
import os
from src.utils.config import SolarSettings
import logging

logger = logging.getLogger(__name__)

class InferenceJob:
    """Solar orchestrator for inference"""
    
    def __init__(self, config: SolarSettings):
        self.config = config
        self.init_mlflow()
        
        # Static arguments
        self.model_name: str = "Solar_MultiHorizon_Forecaster"
        self.alias: str = "champion"
        
    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.alias}"

    def init_mlflow(self) -> None:
            """Tracking MLflow server connexion initialization"""
            try:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_tracking_username.get_secret_value()
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_tracking_password.get_secret_value()
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri.get_secret_value())
            
            except KeyError as e:
                raise ValueError(f"Missing configuration : {e}")
    
    def load_model(self): 
        if hasattr(self, "_model"):
            return self._model

        self._model = mlflow.pyfunc.load_model(self.model_uri)
        return self._model
    
    def _apply_scaling(self, predictions: pd.DataFrame, quantile_factor: float):
        return predictions * quantile_factor
    
    def offset_to_datetime(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Replace offsets by datetime to simplify the output.
        """
        preds = predictions.copy()
        
        base_time = preds.index[0]
        hours = [int(col.replace('+', '').replace('h','')) for col in preds.columns]
        new_cols = [base_time + pd.Timedelta(hours=h) for h in hours]

        preds.columns = new_cols

        return preds

    def inference(
            self, 
            input_dataset: pd.DataFrame,
            loaded_model,
            ) -> pd.DataFrame:
        """
        Inference orchestrator. Load the model from Mlflow and return 24 predictions.
        """
        
        if input_dataset.empty:
            logger.error("input dataset is empty. Abort inference")
        last_row = input_dataset.tail(1)
        predictions = loaded_model.predict(last_row)

        return predictions

    def to_sql_format(
            self,
            predictions: pd.DataFrame) -> pd.DataFrame:
        """Return predictions with correct database format"""

        if predictions.empty:
            raise ValueError("No predictions. Please use .inference() before sql_format.")
        
        df_to_upload = predictions.copy()
        
        # Transformations
        df_to_upload = ( df_to_upload
                        .reset_index(names="predicted_at")
                        .melt(
                            id_vars=["predicted_at"],
                            var_name='forecast_horizon',
                            value_name="predicted_value")
        )

        #Ensure type
        df_to_upload["predicted_at"] = pd.to_datetime(df_to_upload["predicted_at"])
        df_to_upload["forecast_horizon"] = pd.to_datetime(df_to_upload["forecast_horizon"])
        
        return df_to_upload
    

    def run(self, input_dataset: pd.DataFrame, quantile_factor: float):
        """global orchestrator"""
        
        # Init
        loaded_model = self.load_model()
        
        # Predictions
        predictions = self.inference(
            input_dataset=input_dataset,
            loaded_model = loaded_model,
            )
        predictions = self._apply_scaling(
            predictions=predictions, 
            quantile_factor=quantile_factor
            )
        predictions = self.offset_to_datetime(
            predictions=predictions
            )
        predictions = self.to_sql_format(
            predictions=predictions
        )

        return predictions