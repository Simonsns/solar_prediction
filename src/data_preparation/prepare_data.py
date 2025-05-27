# Ce fichier importe les datasets d'intérêts (RTE + Météo) afin de les préparer pour la phase d'exploration
# author : Simon Senegas
# 2025/05/27

# Import libraries

import pandas as pd
import numpy as np
import geopandas as gpd
from fetch_data import fetch_hourly_hist_weather_data
import time
import logging
import warnings

warnings.warn("TODO: La fonction primitive_data_filter n'est pas encore implémentée.", UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

####

def fetch_all_hourly_weather_runs(url: str, start_date: str, end_date: str, 
                                  variables: list, coordinates: gpd.GeoDataFrame) -> list:
    
    """Appelle l'api via l'url (url) entre la date de départ (start_date) et la fin (end_date)
    et renvoie une liste de de données météo de la longueur des coordonnées données (coordinates), avec 
    toutes les variables listées dans (variables) incluses """

    weather_df_list = []
    lon, lat = coordinates.geometry.x, coordinates.geometry.y

    for i in range(len(coordinates)):
        
        df = fetch_hourly_hist_weather_data(url, start_date, end_date, variables, lon[i], lat[i])
        df.columns = [f"{col}_run_{i}" for col in df.columns]
        # TODO: Remplacer le time.sleep par une gestion propre du throttling API (via retry ou async)
        time.sleep(10) 
        weather_df_list.append(df)
    
    return weather_df_list

def separate_central_scenario(weather_df_list: list):
    
    """Separe le scénario central (dernier élément de la liste) des autres dataframes"""
    
    df_central = weather_df_list[-1] # DataFrame scénario central
    df_others = weather_df_list[:-1] # Autres dataframes (ici points de mesures départementaux)
    df_other = pd.concat(df_others, axis=1)
    
    return df_central, df_other

def set_time_index_drop_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dates superflues provenant de l'appel à l'api"""
    
    date_col_index = [col for col in df.columns if col.startswith("date")][0]
    df = df.set_index(date_col_index)
    df = df.drop(columns=[col for col in df.columns if col.startswith("date")])

    return df

def compute_variable_dispersion(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Compute les écarts min-max et la dispersion des différents points de mesure sur 
    les différentes variables listées"""

    df_result = pd.DataFrame(index=df.index)
    
    for var in variables : 
        logging.info(f"{var} en cours de traitement")
        col_filter = [col for col in df.columns if col.startswith(var)]
        data = df[col_filter]
        df_result[f"{var}_delta_minmax"] = (data.max(axis=1) - data.min(axis=1)).round(4)
        df_result[f"{var}_std"] = data.std(axis=1).round(4)

    logging.info("Traitement des variables météorologiques terminé")

    return df_result

def concatenate_weather_production(df_weather: pd.DataFrame, df_dispersion: pd.DataFrame) -> pd.DataFrame:
    """
    Concatène le scénario météo central (df_weather), la dispersion des variables météo (df_dispersion),
    les données de production solaire (df_production), en gérant les agrégations temporelles
    
    TODO : - Ajouter la gestion des agrégations temporelles (ex: daily, hourly resampling)
    - Intégrer les données de production solaire (df_production)"""
    
    date_col_index = [col for col in df_weather.columns if col.startswith("date")][0]
    df_central = df_weather.set_index(date_col_index)
    df_central = pd.merge(df_central, df_dispersion, left_index=True, right_index=True)

    return df_central

def primitive_data_filter(df):
    return NotImplementedError("La fonction primitive_data_filter n'est pas encore implémentée.")

#%%
if __name__ == "__main__":

    coordinates = gpd.read_file("coordinates")
    start_date = "date1"
    end_date = "daet2"
    url = "url"
    variables = "parameters"

    df_weather_list = fetch_all_hourly_weather_runs(url, start_date, end_date, variables, coordinates)
    df_cweather, df_other = separate_central_scenario(df_weather_list)
    df_alternative_weather = set_time_index_drop_date_columns(df_other)
    df_dispersion = compute_variable_dispersion(df_alternative_weather, variables)
    df_central_weather = concatenate_weather_production(df_cweather, df_dispersion)