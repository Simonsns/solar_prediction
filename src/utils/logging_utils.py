import mlflow
import os
from src.models import contracts
from src.models import model_wrappers
import mlflow.lightgbm as lightgbm

### MLFLOW server initialization ### 
def init_mlflow(config: dict, experiment_name: str) -> None:
    """
    MLFlow initializing
    """
    os.environ["MLFLOW_TRACKING_USERNAME"] = config["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config["MLFLOW_TRACKING_PASSWORD"]
    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name=experiment_name)

### LGBM ###
def log_horizon_run(result: contracts.HorizonRunResult) -> None:
    """MLflow logs and artifact"""
    
    # Tags - params - metrics - artifact
    mlflow.set_tags(result.tags_to_log)
    mlflow.log_params(result.params_to_log)
    mlflow.log_metric("best_rmse_val", result.best_rmse)
    lightgbm.log_model(
        result.model,
        name="model",
        signature=result.signature,
        input_example=result.X_train.iloc[:3]
    )

def log_lstm_run(result: contracts.LSTMRunResult) -> None:
    """Logs tags, params, metrics and artifact LSTM on MLflow server"""
    
    mlflow.set_tags(result.tags_to_log)
    mlflow.log_params(result.params_to_log)
    mlflow.log_metric("best_rmse_val", result.best_rmse)
 
    wrapped_model = model_wrappers.SolarLSTMWrapper(
        processor=result.processor,
        model=result.model,
        encoder_features=result.encoder_features,
        decoder_features=result.decoder_features,
        seq_length=result.seq_length,
        output_len=result.output_len,
        device="cpu",
    )
 
    mlflow.pyfunc.log_model(
        name="lstm_seq2seq_model",
        python_model=wrapped_model,
        signature=result.signature
    )