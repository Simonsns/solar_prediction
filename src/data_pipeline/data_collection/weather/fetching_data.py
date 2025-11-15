#%% Librairies
import openmeteo_requests
import pandas as pd
import geopandas as gpd
import requests_cache
from retry_requests import retry
import logging
import time
from tqdm import tqdm
from typing import List
from src.data_pipeline.data_processing.weather.preprocessing import (separate_central_scenario, 
                                                                    set_time_index_drop_date_columns,
                                                                    compute_variable_dispersion, 
                                                                    concatenate_weather_data)
#%%
def fetch_hourly_weather_data(url: str, 
                                   	start_date: str, 
                                	end_date: str, 
                                    variables: list, 
                                    lon: float, 
                                    lat: float) -> pd.DataFrame:
	
	"""Appel à l'API météo. Retourne en sortie un dataframe avec 22 features météorologiques 
      aux coordonnées données.

	Args:
    	url (str): url de l'api
		start_date (str): Date de début des données
		end_date (str): Date de fin des données
        lat (float): latitude du point météorologique étudié
        lon (float): longitude du point météorologique étudié

	Returns:
		hourly_weather_df (pd.DataFrame): DataFrame avec les variables 
        météorologiques horaire sur la fenêtre temporelle donnée
	"""
	# Initialisation du client OPEN-METEO
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 5)
	openmeteo = openmeteo_requests.Client(session = retry_session) # type: ignore
	
	# Toutes les variables sont mentionnées ici 
	params = {
		"longitude": [lon],
		"latitude": [lat],
		"start_date": f'{start_date}',
		"end_date": f'{end_date}',
		"hourly": variables,
	}

	try:
		responses = openmeteo.weather_api(url, params=params)
		response = responses[0]

		logging.info(f"Coordonnées : {response.Latitude()}°N, {response.Longitude()}°E")
		logging.info(f"Altitude : {response.Elevation()} m")
		logging.info("Fuseau horaire : Europe/Paris")

		hourly = response.Hourly()
		time_utc = pd.date_range(
			start=pd.to_datetime(hourly.Time(), unit="s", utc=True), # type: ignore
			end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), # type: ignore
			freq=pd.Timedelta(seconds=hourly.Interval()), # type: ignore
			inclusive="left"
		)

		time_range = time_utc.tz_convert("Europe/Paris") # handmade correction to control data manually
		hourly_data = {"date": time_range}

		for i, var in enumerate(variables):
			hourly_data[var] = hourly.Variables(i).ValuesAsNumpy() # type: ignore    
		
		return pd.DataFrame(data=hourly_data)

	except Exception as e:
		logging.error(f"Erreur lors de la récupération des données météo : {e}")
		raise

#%%
def fetch_all_hourly_weather_runs(url: str, start_date: str, end_date: str, 
                                  variables: list, coordinates: gpd.GeoDataFrame) -> list:
    
    """Appelle l'api via l'url (url) entre la date de départ (start_date) et la fin (end_date)
    et renvoie une liste de de données météo de la longueur des coordonnées données (coordinates), avec 
    toutes les variables listées dans (variables) incluses """

    weather_df_list = []
    lon, lat = coordinates.geometry.x, coordinates.geometry.y

    for i in tqdm(range(len(coordinates))):
        
        df = fetch_hourly_weather_data(url, start_date, end_date, variables, lon[i], lat[i])
        df.columns = [f"{col}_run_{i}" for col in df.columns]
        time.sleep(10) # Gérer l'appel max par secondes
        weather_df_list.append(df)
    
    return weather_df_list
#%%
def _fetch_weather_data(coordinates: gpd.GeoDataFrame,
                        start_date: str,
                        end_date: str,
                        variables: List[str],
                        weather_url: str) -> pd.DataFrame:
    """Pipeline générique de récupération et de traitement météo (historique ou prévisionnelle)."""

    df_weather_list = fetch_all_hourly_weather_runs(weather_url, 
                                                    start_date, 
                                                    end_date,
                                                    variables, 
                                                    coordinates)
    df_cweather, df_other = separate_central_scenario(df_weather_list)
    df_alternative_weather = set_time_index_drop_date_columns(df_other)
    df_dispersion = compute_variable_dispersion(df_alternative_weather, variables)
    df_weather = concatenate_weather_data(df_cweather, df_dispersion)

    return df_weather
#%%
def fetch_historical_weather(production_data:pd.DataFrame, 
                             variables: List[str],
                             coordinates: gpd.GeoDataFrame, 
                             weather_url: str):
    
    """
    Récupère et prépare les données météorologiques historiques correspondant aux données de production.

    La fonction télécharge les données météo horaires pour la période couverte par `production_data`,
    calcule les scénarios centraux et alternatifs, puis assemble un jeu de données complet et aligné
    temporellement avec la production.

    Args:
        production_data (pd.DataFrame): Données historiques de production énergétique, indexées par heure.
        variables (List[str]): Liste des variables météorologiques à récupérer (ex. température, irradiance, vent).
        coord_path (str): Chemin vers le fichier des coordonnées géographiques (format supporté par GeoPandas).
        weather_url (str): URL de la source des données météorologiques historiques.

    Returns:
        pd.DataFrame: Données météorologiques historiques formatées et alignées sur la période de production.

    Raises:
        AssertionError: Si la longueur des données météo ne correspond pas à celle de `production_data`.
        FileNotFoundError: Si le fichier de coordonnées spécifié par `coord_path` est introuvable.
        ValueError: Si les données météo téléchargées sont incomplètes ou corrompues.
    """
    # Initialisation
    start_date = production_data.index[0].strftime("%Y-%m-%d")
    end_date = production_data.index[-1].strftime("%Y-%m-%d")

    # Données historiques
    logging.info("[INIT] Récupération des données météos historiques")
    df_weather = _fetch_weather_data(coordinates=coordinates, 
                                    start_date=start_date, 
                                    end_date=end_date,
                                	variables=variables, 
                                    weather_url=weather_url)
    
    df_weather = df_weather.loc[production_data.index[0]:production_data.index[-1]]

    # Test
    assert len(df_weather) == len(production_data), "Les longueurs météo et production diffèrent."
    logging.info("[END] Données météos historiques stockées")
    
    return df_weather

def fetch_forecast_weather(production_data: pd.DataFrame, 
                            variables: List[str],
							coordinates: gpd.GeoDataFrame,
                            len_prev: int,  
                            forecast_weather_url: str):
    """
    Récupère et prépare les données météorologiques prévisionnelles pour une période donnée.

    La fonction télécharge les données de prévision horaire, calcule les scénarios centraux 
    et alternatifs, puis assemble un jeu de données prêt à être utilisé pour la modélisation 
    de la production énergétique.

    Args:
        production_data (pd.DataFrame): Données historiques de production énergétique.
        variables (List[str]): Liste des variables météorologiques à récupérer (ex. température, irradiance, vent).
        len_prev (int): Longueur de la prévision en heures.
        coord_path (str): Chemin vers le fichier contenant les coordonnées géographiques.
        forecast_weather_url (str): URL de la source des données météorologiques prévisionnelles.

    Returns:
        pd.DataFrame: Données météorologiques prévisionnelles formatées et alignées temporellement.
    
    Raises:
        AssertionError: Si la longueur des données prévisionnelles obtenues ne correspond pas à `len_prev`.
        FileNotFoundError: Si le fichier de coordonnées spécifié par `coord_path` est introuvable.
        ValueError: Si les données météo récupérées sont invalides ou manquantes pour la période demandée.
    """
    # Forecast 
    forecast_start = production_data.index[-1] + pd.Timedelta(hours=1)
    forecast_end = forecast_start + pd.Timedelta(hours=len_prev - 1)

    # Recherche des données prévisionnelles
    logging.info("[INIT] Récupération des données météos prévisionnelles")
    df_weather = _fetch_weather_data(coordinates,
          							forecast_start.strftime("%Y-%m-%d"),
                                    forecast_end.strftime("%Y-%m-%d"),
                                    variables,
                                    forecast_weather_url)

    # Reshape and testing
    df_weather = df_weather.loc[df_weather.index >= forecast_start]
    df_weather = df_weather.head(len_prev)
    
    # Test
    assert len(df_weather) == len_prev, "La longueur des prévisions ne correspond pas à len_prev"
    logging.info("[END] Données météos historiques stockées")

    return df_weather