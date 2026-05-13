# Utils
import os
import logging
from typing import Any, Dict, List, Tuple

# Libraries
from datetime import datetime
import pandas as pd
import numpy as np
import optuna
import mlflow
from mlflow.tracking import MlflowClient
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from optuna.pruners import MedianPruner

# Custom modules
from src.data_pipeline import processors
from src.utils.config import SolarSettings
from src.models import contracts, model_wrappers
from src.utils import metrics
logger = logging.getLogger(__name__)

class SolarTrainingOrchestrator:
    """
    End-to-end orchestrator to train multi-horizons solar forecasting.
    Handle HP optimization, horizon training and MLFlow serialization.
    """

    def __init__(
            self,
            config: SolarSettings,
            meteo_features: List[str],
            selector_parameters: Dict[str, Any],
            experiment_name: str = "LGBM_Direct_Output",
            n_horizons: int = 24,
            n_cv_splits: int = 5,
            num_trials: int = 30,
            random_state: int = 42,
    ):
        # 1 - Config injection
        self.config = config
        self.meteo_features = meteo_features
        self.selector_parameters = selector_parameters

        # 2 - Experiment parameters
        self.experiment_name = experiment_name
        self.n_horizons = n_horizons
        self.n_cv_splits = n_cv_splits
        self.num_trials = num_trials
        self.random_state = random_state

        # 3 - Internal state
        self.models_dict: Dict[str, Pipeline] = {}
        self._is_fitted = False
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Tracking MLflow server connexion initialization"""
        try:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_tracking_username.get_secret_value()
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_tracking_password.get_secret_value()
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri.get_secret_value())
            mlflow.set_experiment(experiment_name=self.experiment_name)
            logger.info(f"Initialized MLflow on experiment: {self.experiment_name}")
        
        except KeyError as e:
            raise ValueError(f"Missing configuration : {e}")
        
    def _prepare_horizon_data(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            horizon: int,
            ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features with temporal shift"""
        
        y_shifted = y.shift(-horizon).dropna()
        X_aligned = X.loc[y_shifted.index]
        
        return X_aligned, y_shifted
    
    def _optimize_hyperparameters(
            self, 
            X: pd.DataFrame,
            y: pd.Series,
            current_selector_parameters: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Any]]:
        """Launch Optuna study to fin best hyperparameters (LGBM model)"""

        # Folds
        tscv = TimeSeriesSplit(gap=0, n_splits=self.n_cv_splits)
        
        # Processor fit
        selector_processor = processors.SolarDataProcessor(meteo_features=self.meteo_features)
        selector_processor.fit(X, y)
        X_scaled = selector_processor.transform(X=X)
        y_scaled = selector_processor.transform_y(y=y)

        # Selector fit
        selector = processors.LGBMFeatureSelector(**current_selector_parameters)
        selector.fit(X_scaled, y_scaled) 
        # NOTE: selector fitted on full X before CV folds to speed up HP search.
        # Introduces minor leakage — acceptable trade-off for training efficiency.

        # Static 
        folds_data = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_processor = processors.SolarDataProcessor(meteo_features=self.meteo_features)
            fold_processor.fit(X_train, y_train)

            folds_data.append({
                'X_train': selector.transform(fold_processor.transform(X_train)),
                'y_train': fold_processor.transform_y(y_train),
                'X_val': selector.transform(fold_processor.transform(X_val)),
                'y_val_real': y_val, #MWh for final comparison
                'proc': fold_processor
            })

        def objective_lightgbm(trial) -> np.float64 :
            
            # Init
            params = {
                "num_leaves" : trial.suggest_int("num_leaves", 10, 100),
                "learning_rate" : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth" : trial.suggest_int("max_depth", 1, 30),
                "n_estimators" : trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples" : trial.suggest_int("min_child_samples", 10, 50),
                "verbosity" : -1,
                "random_state": 42
            }
            scores = []

            for fold in folds_data:
                model = lgb.LGBMRegressor(**params)
                model.fit(fold['X_train'], fold['y_train'])
                y_pred_norm = model.predict(fold['X_val'])

                # Denormalization
                y_pred = fold['proc'].inverse_transform_y(y_pred_norm, fold['y_val_real'].index)
                scores.append(metrics.rmse(fold['y_val_real'], y_pred))
            
            return np.mean(scores) # type: ignore
        
        # Recherche des HP et prédiction
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        study.optimize(objective_lightgbm, n_trials=self.num_trials, n_jobs=4)

        return study.best_value, study.best_params
        
    def _fit_single_horizon(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            horizon: int,
        ) -> contracts.HorizonRunResult:
        """Optimize and trains the final model for a given horizon"""
        X_h, y_h = self._prepare_horizon_data(X=X, y=y, horizon=horizon)

        current_selector_params = self.selector_parameters.copy()
        if horizon <= 3:
            set_name = "short"
            current_selector_params["horizons"] = [1, 3]
        elif horizon <= 12:
            set_name = "mid"
            current_selector_params["horizons"] = [3, 6, 12]
        else:
            set_name = "long"
            current_selector_params["horizons"] = [12, 24]
        
        logger.info(f"Start horizon optimization +{horizon}h ({set_name})")
        best_rmse, best_params = self._optimize_hyperparameters(X_h, y_h, current_selector_params)

        # Final training
        processor = processors.SolarDataProcessor(meteo_features=self.meteo_features)
        processor.fit(X_h, y_h)
        X_scaled = processor.transform(X_h)
        y_scaled = processor.transform_y(y_h)

        selector = processors.LGBMFeatureSelector(**current_selector_params)
        selector.fit(X_scaled, y_scaled)
        X_final_input = selector.transform(X_scaled)

        final_model = lgb.LGBMRegressor(**best_params, verbosity=-1, random_state=self.random_state)
        final_model.fit(X_final_input, y_scaled)

        pipeline = Pipeline([
            ('processor', processor),
            ('selector', selector),
            ('model', final_model)
        ])

        return contracts.HorizonRunResult(
            horizon=horizon,
            set_name=set_name,
            best_rmse=best_rmse,
            best_params=best_params,
            model=pipeline,
            X_train=X_h
        )

    def run_training_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
        """
        Main entrypoint. Multihorizons iterations, launch trainings and log results on MLFlow (interlocking run)
        """

        logger.info(f"Pipeline training launch for {self.n_horizons} horizons")

        # 1 - Global run
        with mlflow.start_run(run_name="MultiHorizon_Training_Pipeline") as parent_run:
            for horizon in range(1, self.n_horizons + 1):
                result = self._fit_single_horizon(X_train, y_train, horizon)
                
                # Registry model for metamodel
                self.models_dict[f'+{horizon}h'] = result.model
                
                # Enfant log run
                with mlflow.start_run(run_name=f"+{horizon}h_{result.set_name}", nested=True):
                    mlflow.set_tags(result.tags_to_log)
                    mlflow.log_params(result.params_to_log)
                    mlflow.log_metric("best_rmse_val", result.best_rmse)
                    
                    mlflow.sklearn.log_model( # type: ignore
                        sk_model=result.model,
                        artifact_path="model",
                        signature=result.signature,
                        input_example=result.X_train.iloc[:3]
                    )

        # 2 - Returns      
        self._is_fitted = True
        logger.info("[SUCCESS] Training pipeline terminated.")
        return self.models_dict
    
    def package_and_log_meta_model(self, X_sample: pd.DataFrame, run_name: str = "LGBM_DirectPrediction_J+1") -> None:
        """
        Wraps all the pipelines int the LGBMwraapper and save on MLflow server as a unique artifact.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted. Use .fit() before MLflow log.")

        date = datetime.now().strftime("%Y-%m-%d")
        artifact_path = f"{date}_wrapped_denormalized_lgbm_model"
        meta_model = model_wrappers.MultiHorizonLGBMWrapper(
            models_dict=self.models_dict, 
            num_horizons=self.n_horizons
            )
        
        # Unique name
        model_name = "Solar_MultiHorizon_Forecaster"

        # Specific run (out of deployment)
        with mlflow.start_run(run_name=run_name):
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=meta_model,
                input_example=X_sample.iloc[:3],
                registered_model_name=model_name
            )

            client = MlflowClient()
            model_version = model_info.registered_model_version

            if model_version is None:
                raise RuntimeError("MLflow model registration failed: no version returned.")
            
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(model_version)
            )

        logger.info(f" {model_version} model version '{model_name}' tracked as 'champion'.")
        logger.info("[SUCCESS] MetaModel PyFunc wrapped, logged and aliased on Mlflow.")