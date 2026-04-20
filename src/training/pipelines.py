# Pipeline
import pandas as pd
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn

# Utils
import logging 
import mlflow
from typing import Optional, List, Dict, Any

# Functions
from src.data_pipeline import data_preparation
from src.training import engine
from src.models import contracts
from models import architectures
from src.utils import logging_utils


def training_multioutput_lgbm_model(X: pd.DataFrame, 
                                    y: pd.Series, 
                                    meteo_features: List[str],
                                    n_horizons: int, 
                                    n_cv_splits: int, 
                                    num_trials: Optional[int|None],
                                    feature_sets: Dict[str, List[str]]) -> Dict[str, Pipeline]:
    
    """Train n horizons LGBM and log params, metrics and tags on cloud MLFlow server.

    Args:
        X (pd.DataFrame): Train dataset
        y (pd.Series): Target feature
        n_horizons (int): Horizon number to train
        n_cv_splits (int): Split nomber for time series cross validation
        num_trials (Optional[int | None]): Trial Optuna number
    """

    models_dict = {}
    for horizon in range(1, n_horizons+1):
        logging.info(f'--- Training horizon t+{horizon} ---')
        
        X_h, y_h = data_preparation.prepare_horizon_data(
            X=X, 
            y=y, 
            horizon=horizon
        )
        
        # Features horizon group
        if horizon <= 3:
            set_name = "short"
        elif horizon <= 12:
            set_name = "mid"
        else:
            set_name = "long"

        # Training 
        features = feature_sets[set_name]
        model, best_rmse, best_params = engine.fit_best_model(
            X=X_h[features], y=y_h,
            meteo_features=meteo_features,
            selected_features=features,
            n_cv_splits=n_cv_splits,
            num_trials=num_trials
        )

        # Save for wrap
        models_dict[f'h+{horizon}'] = model

        # MLFlow
        logging.info(f'Best model fitted on {set_name}- Data contract loading')
        result = contracts.HorizonRunResult(
            horizon=horizon,
            set_name=set_name,
            best_rmse=best_rmse,
            best_params=best_params,
            model=model,
            X_train=X_h[features]
        )

        with mlflow.start_run(run_name=f"h+{horizon}_{set_name}", nested=True):
            logging_utils.log_horizon_run(result=result)
    
    return models_dict

def run_lstm_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meteo_features: List[str],
    encoder_features: List[str],
    decoder_features: List[str],
    lstm_params: Dict[str, Any],
) -> None:
    
    """Complete pipeline LSTM : DataLoaders → training → MLflow.
 
    Call train_seq2seq() then log on MLflow server.
    """
    seq_length = lstm_params["seq_length"]
    output_len = lstm_params["output_len"]
    batch_size = lstm_params["batch_size"]
    stride     = lstm_params["stride"]
    num_epochs = lstm_params["num_epochs"]
 
    # 1. Préparation données
    train_dl, val_dl, processor, X_ex, y_past_ex = data_preparation.prepare_lstm_data(
        X=X_train,
        y=y_train,
        meteo_features=meteo_features,
        encoder_features=encoder_features,
        decoder_features=decoder_features,
        seq_length=seq_length,
        output_len=output_len,
        batch_size=batch_size,
        stride=stride,
    )
 
    # 2. Instanciation modèle
    past_features_length   = len(encoder_features) + 1
    future_features_length = len(decoder_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = architectures.Seq2seq( 
        past_features_length=past_features_length,
        future_features_length=future_features_length,
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        output_len=output_len,
        dropout=lstm_params["dropout"],
    ).to(device)
 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
 
    # 3. Training (déjà dans le notebook)
    train_losses, val_losses = engine.train_seq2seq(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )
    best_rmse = min(val_losses)
 
    # 4. Data contract
    result = contracts.LSTMRunResult(
        num_epochs=num_epochs,
        seq_length=seq_length,
        output_len=output_len,
        model=model.to("cpu"),
        best_rmse=best_rmse,
        best_params=lstm_params,
        encoder_features=encoder_features,
        decoder_features=decoder_features,
        processor=processor,
    )
 
    # 5. MLflow
    with mlflow.start_run(run_name="LSTM_Seq2Seq_J+1"):
        logging_utils.log_lstm_run(result=result)