# Fichier contenant l'ensemble des fonctions utilisables de façon commune
# author : simon.senegas
# last_date : 12/11/2025
#%%
import pandas as pd
import geopandas as gpd
import logging
import numpy as np
from typing import Dict
#%%
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

def lagged_ma_feature_encoding(df: pd.DataFrame, 
                               feature_list: list[str], 
                               lag_list: list[int]) -> pd.DataFrame:
    
    """Prends en entrée un DataFrame df indexée temporellement avec une liste de features (feature_list) et de lags (lag_list), et renvoie le dataframe df munie des features laggées et
    d'une moyenne mobile sur les mêmes lags.

    Args:
        df (pd.DataFrame): DataFrame indexé temporellement
        feature_list (list[str]): Features (noms de colonnes du dataframe) que l'on veut retarder ou lisser.
        lag_list (list[int]): Liste de lags à imputer

    Returns:
        pd.DataFrame: Retourne le DataFrame munie des features laggées et les moyennes mobiles associées
    """

    for col in feature_list:
        for lag in lag_list:
            #Feature laggée
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            #Moyenne mobile sur lag période
            if col != "solaire":
                df[f"{col}_ma_{lag}"] = df[col].rolling(window=lag).mean()
    
    return df

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
# %%
