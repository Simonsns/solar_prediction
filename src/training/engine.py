# Training logic file 
# SARIMAX, LGBM and LSTM training functions are stocked here

#Data
import pandas as pd
import numpy as np
from math import inf
from XXX import SolarDataProcessor

# ML - DL
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.utils.metrics import rmse

# Utils
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
import traceback

############################################################################ SARIMAX ####################################################################

def train_sarimax(X: pd.DataFrame, y: pd.Series,  nb_cv_splits: int, num_trials: Optional[int|None]) -> tuple:
    
    """Entrainement d'un modèle SARIMAX avec nombre d'essais pour optimisation.
    Retourne la quantification de son erreur (best_rmse) et ses hyperparamètres 
    optimaux (dict_best_params) sur un dataset de validation (X_val, y_val)

    Args:
        X_train (pd.DataFrame): Dataset d'entrainement
        y_train (pd.Series): Variable cible d'entrainement
        X_val (pd.DataFrame): Dataset de validation
        y_val (pd.Series): Variable cible de validation

    Returns:
        best_rmse (float), dict_best_params (Dict) : Erreur (best_rmse) 
        et ses hyperparamètres optimaux (dict_best_params)
    """
    # Initialisation
    folds = TimeSeriesSplit(n_splits=nb_cv_splits)

    def objective_sarimax(trial) -> np.float64 :
        """Prend en entrée un set d'hyperparamètres SARIMAX issus du sampler d'Optuna, 
        et retourne le RMSE associé.

        Args:
            trial : Set d'hyperparamètres SARIMAX

        Returns:
            rmse (np.float64): Listes des RMSE de la CV du modèle
        """
        rmses = []

        # hyperparamètres
        order = (trial.suggest_int("p", 0, 1),
            trial.suggest_int("d", 0, 1),
            trial.suggest_int("q", 0, 1))

        seasonal_order = (
            trial.suggest_int("P", 0, 1), 
            trial.suggest_int("D", 0, 1),
            trial.suggest_int("Q", 0, 1),
            24, #Saisonnalité fixe
        )
        
        for train_idx, val_idx in folds.split(X):
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Training
            try:
                logging.info(f"Lancement de l'entrainement sur les paramètres : {order}\n{seasonal_order}")
                model = SARIMAX(endog=y_train, 
                                exog=X_train, 
                                order = order, 
                                seasonal_order = seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                fitted_model = model.fit(disp=False)
                logging.info(f"Prédiction du modèle : {order}\n{seasonal_order}")
                
                # Prédictions sur validation
                prediction_interval = fitted_model.get_prediction(start=y_val.index[0],
                                                          end=y_val.index[-1],
                                                          exog=X_val)
                predictions = prediction_interval.predicted_mean
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                rmses.append(rmse)
            
            except Exception as e:
                logging.info("Echec de l'entrainement du modèle :", e)
                logging.debug("Détails complets :\n%s", traceback.format_exc())
                return inf

        return np.mean(rmses)
    
    # Recherche des HP et prédictions
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_sarimax, n_trials=num_trials, n_jobs=1)
    
    except Exception as e:
        logging.error(
        "Échec de la recherche d’HP. Paramètres : n_trials=%d, n_jobs=%d. Erreur : %s",
        num_trials, 4, str(e),
        exc_info=True)
        raise
    
    return study.best_value, study.best_params

#################################################################### LGBM ############################################################################

def prepare_horizon_data(X: pd.DataFrame, y: pd.Series, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return shift + alignment for a given horizon"""
    
    # Init
    y_shifted = y.shift(-horizon).dropna()
    X_aligned = X.loc[y_shifted.index]

    return X_aligned, y_shifted

def train_lightgbm(X: pd.DataFrame, 
                   y: pd.Series, 
                   meteo_features: List[str],
                   nb_cv_splits: int,
                   num_trials: Optional[int|None]
                   ) -> Tuple[float, Dict[Any, Any]]:
    
    """Entrainement d'un modèle LightGBM avec nombre d'essais pour optimisation.
    Retourne la quantification de son erreur (best_rmse) et ses hyperparamètres 
    optimaux (dict_best_params) sur un dataset de validation (X_val, y_val)

    Args:
        X_train (pd.DataFrame): Dataset d'entrainement
        y_train (pd.Series): Variable cible d'entrainement
        X_val (pd.DataFrame): Dataset de validation
        y_val (pd.Series): Variable cible de validation

    Returns:
        best_rmse (float), dict_best_params (Dict) : Erreur (best_rmse) 
        et ses hyperparamètres optimaux (dict_best_params)
    """

    # Métriques scorer et folds
    tscv = TimeSeriesSplit(gap=0, n_splits=nb_cv_splits)

    def objective_lightgbm(trial) -> np.float64 :
        """Prend en entrée un set d'hyperparamètres LightGBM issus du sampler d'Optuna, 
        et retourne le RMSE associé.

        Args:
            trial : Set d'hyperparamètres LightGBM

        Returns:
            rmse (np.float64): Racine carrée de l'erreur quadratique moyenne du modèle
        """
        
        # HP
        params = {
            "num_leaves" : trial.suggest_int("num_leaves", 10, 100),
            "learning_rate" : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth" : trial.suggest_int("max_depth", 1, 30),
            "n_estimators" : trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples" : trial.suggest_int("min_child_samples", 10, 50),
            "verbosity" : -1,
            "random_state": 42
        }

        
        # Cross validation manuelle car cross_val_score ne gère pas inverse_transform de y
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Model fit avec les bonnes valeurs
            # y normalization outer : La pipeline ne transforme pas la target
            final_processor = SolarDataProcessor(meteo_features=meteo_features)
            final_processor.fit(X_train, y_train)
            X_train_scaled = final_processor.transform(X=X_train)
            y_train_scaled = final_processor.transform_y(y=y_train)

            # Model fit
            final_model = lgb.LGBMRegressor(**params)
            final_model.fit(X_train_scaled, y_train_scaled)
            
            # Pipeline already fitted
            pipeline = Pipeline([
            ('processor', final_processor),
            ('model', final_model)
        ])
        
            # Fitting total
            norm_y_pred = pipeline.predict(X_val)

            # On compare des MWh avec des MWh
            y_pred = pipeline.named_steps['processor'].inverse_transform_y(norm_y_pred, y_val.index) 
            
            scores.append(rmse(y_true=y_val, y_pred=y_pred))

        return np.mean(scores)
    
    # Recherche des HP et prédiction
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                pruner=MedianPruner(
                                    n_startup_trials=5, 
                                    n_warmup_steps=3)
                                )
    
    study.optimize(objective_lightgbm, n_trials=num_trials, n_jobs=-1)

    return study.best_value, study.best_params

def fit_best_model(X: pd.DataFrame, 
                   y: pd.Series, 
                   meteo_features: List[str],
                   selected_features: List[str],
                   n_cv_splits: int, 
                   num_trials: Optional[int|None]) -> Tuple[Pipeline, float, dict]:
    """
    Return (model, best_rmse, best_params).
    """
    
    # HP Searching
    best_rmse, best_params = train_lightgbm(
        X=X, y=y,
        meteo_features=meteo_features,
        nb_cv_splits=n_cv_splits,
        num_trials=num_trials,
    )

    # Model fit avec les bonnes valeurs
    final_processor = SolarDataProcessor(
        meteo_features=meteo_features, 
        selected_features=selected_features)
    
    final_processor.fit(X, y)
    X_train_scaled = final_processor.transform(X=X)
    y_train_scaled = final_processor.transform_y(y=y)
    
    # Fit model
    final_model = lgb.LGBMRegressor(**best_params, verbosity=-1, random_state=42)
    final_model.fit(X_train_scaled, y_train_scaled)

    # Pipeline already fitted
    pipeline = Pipeline([
    ('processor', final_processor),
    ('model', final_model)
    ])

    return pipeline, best_rmse, best_params

####################################################################### LSTM SEQ2SEQ ###########################################################################

@dataclass
class EarlyStopping:
    """Early stopping avec patience sur la val loss."""

    patience:  int     = 10
    min_delta: float   = 1e-4
    best_loss:   float = field(default=float("inf"), init=False)
    counter:     int   = field(default=0,            init=False)
    should_stop: bool  = field(default=False,        init=False)

    def step(self, val_loss: float) -> bool:
        """Retourne True if val_loss is stagnating"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
    
def train_seq2seq(model, 
                       train_dataloader: DataLoader, 
                       val_dataloader: DataLoader,
                       criterion,
                       optimizer, 
                       device,
                       num_epochs: int,
                       patience: int= 10) -> Tuple[List[float], List[float]]:
    """
    Trains a seq2seq model and evaluates it on a validation set each epoch.

    Args:
        model (torch.nn.Module): Seq2seq model.
        train_dataloader (DataLoader): Training data loader.
        val_dataloader (DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device (torch.device): CPU or GPU device.
        num_epochs (int): Number of training epochs.

    Returns:
        tuple: (train_losses, val_losses), lists of RMSE per epoch.
    """
    
    #Init 
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []

    logging.info(f"Début de l'entrainement sur {num_epochs} sur {device}")
    
    for epoch in range(num_epochs):
        
        # Training

        model.train()
        running_loss = 0.0

        for X_past, X_future, Y_target in train_dataloader:
            X_past, X_future, Y_target = X_past.to(device), X_future.to(device), Y_target.to(device)

            #Mise à zéro gradient
            optimizer.zero_grad()

            # Forward et perte
            Y_pred = model(X_past, X_future)
            loss = criterion(Y_pred, Y_target)

            # Rétropropagation et optimisation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Statistiques
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(np.sqrt(avg_train_loss))
        
        # Eval et validation dataset

        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for X_past_val, X_future_val, Y_target_val in val_dataloader:
                X_past_val, X_future_val, Y_target_val = X_past_val.to(device), X_future_val.to(device), Y_target_val.to(device)
                
                Y_pred_val = model(X_past_val, X_future_val)
                loss_val = criterion(Y_pred_val, Y_target_val)

                validation_loss += loss_val.item()
                
        avg_val_loss = validation_loss / len(val_dataloader) # type: ignore
        val_losses.append(np.sqrt(avg_val_loss))

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train RMSE: {np.sqrt(avg_train_loss):.6f} | Val RMSE: {np.sqrt(avg_val_loss):.6f}")

        if early_stopping.step(avg_val_loss):
            logging.info(f"Early stopping à l'epoch {epoch+1} — best val loss : {early_stopping.best_loss:.6f}")
            break
        
    return train_losses, val_losses