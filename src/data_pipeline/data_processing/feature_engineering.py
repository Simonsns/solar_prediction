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

def encoding_multihorizons_features(df: pd.DataFrame, 
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

    for col in feature_list:
        
        for lag in lag_list:
            new_cols[f"{col}_lag_t-{lag}"] = df[col].shift(lag) #Feature laggée

        for window in window_list:
            new_cols[f"{col}_ma_{window}"] = df[col].rolling(window=window, min_periods=1).mean() #Moyenne mobile
            new_cols[f"{col}_volatility_{window}"] = df[col].rolling(window=window, min_periods=1).std() #Volatilité sur window
            new_cols[f"{col}_ramp_{window}"] = df[col].shift(1) - df[col].shift(window+1) #Ramp sur window

    df_out = pd.concat([df, pd.DataFrame(data=new_cols)], axis=1)        
    
    return df_out

def full_raw_inference_dataset(production_data: pd.DataFrame, 
                        historical_weather: pd.DataFrame, 
                        forecast_weather: pd.DataFrame, 
                        sum_capacity: np.float64) -> pd.DataFrame: 
    """
    Construit un jeu de données complet pour l’inférence en combinant les données de production,
    les données météorologiques historiques et les prévisions météorologiques.

    La fonction fusionne d'abord les données de production et de météo historique selon leur index temporel,
    puis concatène les prévisions météo pour produire un ensemble continu prêt à être feature engineer.

    Args:
        production_data (pd.DataFrame): Données historiques de production énergétique.
        historical_weather (pd.DataFrame): Données météorologiques historiques alignées temporellement.
        forecast_weather (pd.DataFrame): Données météorologiques prévisionnelles.

    Returns:
        pd.DataFrame: Jeu de données complet fusionnant historique et prévisions, indexé temporellement.
    """

    production_data.loc[:, "solaire"] = production_data.loc[:, "solaire"]/sum_capacity
    full_data = pd.merge(production_data, historical_weather, left_index=True, right_index=True)
    full_data = pd.concat([full_data, forecast_weather], axis=0).sort_index()
        
    return full_data

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

    for col in feature_list:
        
        for lag in lag_list:
            new_cols[f"{col}_t+{lag}"] = df[col].shift(-lag)                          #Feature laggée
        
        for delta in delta_list:   
            new_cols[f"{col}_delta_t+{delta}_t"] = df[col].shift(-delta) - df[col]    # delta entre lag et valeur actuelle

    return pd.DataFrame(data=new_cols, index=df.index)

def transform_pipeline(raw_inference_data: pd.DataFrame,
                       timeframe_dict: Dict,
                       lag_list: List[int],
                       window_list: List[int],
                       lagged_feature_list: List[str],
                       central_scenario: int) -> pd.DataFrame:
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
    logging.info("[INIT] Pipeline de transformation en cours")
    
    try:
        # Renommage
        df = col_scenario_rename(raw_inference_data, central_scenario)
        
        # Encodage cyclique
        df = cyclical_features_encoding(df, timeframe_dict)
        logging.info(f"Encodage cyclique effectué pour : {list(timeframe_dict.keys())}")

        # Variables laggées 
        df = encoding_multihorizons_features(df, lagged_feature_list, lag_list, window_list)
        logging.info(f"Variables laggées créées pour les lags : {lag_list}")

        # Nettoyage final
        inference_data = df.copy()
        inference_data = inference_data.dropna()
        inference_data.index = inference_data.index.rename("date_heure")

        return inference_data
    
    except Exception as e :
        logging.exception("[ERROR] Interruption de la pipeline de transformation")
        raise e