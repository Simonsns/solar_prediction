# Fichier contenant l'ensemble des fonctions utilisables de façon commune
# author : simon.senegas
# last_date : 07/12/2025

import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Iterable

def col_scenario_rename(df: pd.DataFrame, run_filter: int) -> pd.DataFrame:
    """Retourne un dataframe avec les colonnes sans chiffre "_run_X"""
    
    df.columns = df.columns.str.replace(f"_run_{run_filter}", "")
    
    return df

def filter_covariable(df: pd.DataFrame, filter_col_list: list[str]) -> pd.DataFrame:
    """Filtre les covariables selon une liste données"""
    return df[[col for col in df.columns if any(f in col for f in filter_col_list)]]

def cyclical_features_encoding(X: pd.DataFrame, timeframe_dict: Dict[str, int]) -> pd.DataFrame:
    
    """Encode les features cycliques dans un dataframe (df) muni d'un index temporel, par les features données dans le dictionnaire (timeframe_dict)

    Args:
        df (pd.DataFrame): DataFrame muni d'un TimeIndex
        timeframe_dict (dict): Dictionnaire muni du nom de la timeframe et de la saisonnalité associée : {"hour": 24}

    Returns:
        pd.DataFrame: DataFrame muni des colonnes temporelles cycliques (clé dictionnaire+"_cos", "clé dictionnaire"+"_sin")
    """
    df = X.copy()

    for key, item in timeframe_dict.items():
        
        df[key] =  getattr(df.index, key)
        df[f"{key}_sin"] = np.round(np.sin(2 * np.pi * df[key]/item), 5)
        df[f"{key}_cos"] = np.round(np.cos(2 * np.pi * df[key]/item), 5)

    df = df.drop(columns=list(timeframe_dict.keys()))

    return df

def prepare_past_features(df: pd.DataFrame, 
                        target: pd.DataFrame,
                        feature_list: Iterable[str], 
                        lag_list: list[int], 
                        window_list: list[int]) -> pd.DataFrame:
    
    """Prends en entrée un DataFrame df indexée temporellement avec une liste de features (feature_list) 
    et de lags (lag_list), et renvoie le dataframe df munie des features laggées, volatilité, ramp 
    et d'une moyenne mobile selon des fenêtres (window_list).

    Args:
        df (pd.DataFrame): DataFrame indexé temporellement
        feature_list (Iterable[str]): Features (noms de colonnes du dataframe) que l'on veut retarder ou lisser.
        lag_list (list[int]): Liste de lags à imputer
        window_list (list[int]): Liste de fenêtre pour les variables glissantes (ramp, moyenne, volatilité)

    Returns:
        pd.DataFrame: Retourne le DataFrame munie des features laggées et les moyennes mobiles associées
    """

    new_cols = {}
    past_data = pd.merge(df, target, left_index=True, right_index=True)

    for col in feature_list:
        
        for lag in lag_list:
            new_cols[f"{col}_lag_t-{lag}"] = past_data[col].shift(lag) #Feature laggée

        for window in window_list:
            new_cols[f"{col}_ma_{window}"] = past_data[col].shift(1).rolling(window=window, min_periods=1).mean() #Moyenne mobile
            new_cols[f"{col}_volatility_{window}"] = past_data[col].shift(1).rolling(window=window, min_periods=1).std() #Volatilité sur window
            new_cols[f"{col}_ramp_{window}"] = past_data[col].shift(1) - past_data[col].shift(window+1) #Ramp sur window
    
    return pd.concat([past_data, pd.DataFrame(data=new_cols)], axis=1)       

def prepare_forecast_features(df: pd.DataFrame, 
                                    feature_list: Iterable[str], 
                                    lag_list: list[int],
                                    delta_list: list[int]) -> pd.DataFrame:
    
    """Prends en entrée un DataFrame df indexée temporellement avec une liste de features (feature_list) 
    et de lags (lag_list), et renvoie le dataframe df munie des features laggées et les deltas des 
    features prévisionnelles.

    Args:
        df (pd.DataFrame): DataFrame indexé temporellement
        feature_list (Iterable[str]): Features (noms de colonnes du dataframe) que l'on veut retarder ou lisser.
        lag_list : liste d'entiers représentant les horizons pour lesquels créer des lags et deltas.

    Returns:
        pd.DataFrame: Retourne le DataFrame avec exclusivement les features laggées et les deltas associés
    """

    new_cols = {}
    df_copy = df.copy()

    for col in feature_list:
        
        for lag in lag_list:
            new_cols[f"{col}_t+{lag}"] = df_copy[col].shift(-lag)                          #Feature laggée
        
        for delta in delta_list:   
            new_cols[f"{col}_delta_t+{delta}_t"] = df_copy[col].shift(-delta) - df_copy[col]    # delta entre lag et valeur actuelle

    return pd.concat([df, pd.DataFrame(data=new_cols)], axis=1)


def transform_pipeline(inference_data: pd.DataFrame,
                       timeframe_dict: Dict) -> pd.DataFrame:
    """Pipeline de features engineering temporels. Retourne le dataset prêt pour l'inférence du modèle.

    Args:
        raw_inference_data (pd.DataFrame): Dataset en sortie de la phase d'extraction
        timeframe_dict (dict): Dictionnaire permettant de choisir les variables encodées cycliquement 
            (ex : TIMEFRAME_DICT = {"month": 12, "hour": 24})
        lag_list (list): Liste des décalages temporels (lags) appliqués aux variables laggées.
        lagged_feature_list (list):  Liste des noms de variables sur lesquelles appliquer les lags.
        central_scenario (int): Scénario central (id des coordonnées d'études)

    Returns:
        inference_data (pd.DataFrame): Dataset prêt pour l'inférence
    """
    
    # Init
    df = inference_data.copy()
    
    try: 
        # Encodage cyclique
        df = cyclical_features_encoding(df, timeframe_dict)
        logging.info(f"Encodage cyclique effectué pour : {list(timeframe_dict.keys())}")

        # Nettoyage final
        df.index = df.index.rename("date_heure")

        return df
    
    except Exception as e :
        logging.exception("[ERROR] Interruption de la pipeline de transformation")
        raise e