#%% Librairies
import requests
import openmeteo_requests
import pandas as pd
import geopandas as gpd
import requests_cache
from retry_requests import retry
import json
import logging
import time
from io import BytesIO
import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#%%
def fetch_hourly_hist_weather_data(url: str, 
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
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
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

    for i in range(len(coordinates)):
        
        df = fetch_hourly_hist_weather_data(url, start_date, end_date, variables, lon[i], lat[i])
        df.columns = [f"{col}_run_{i}" for col in df.columns]

        # TODO: Remplacer le time.sleep par une gestion propre du throttling API (via retry ou async)
        time.sleep(10) 
        weather_df_list.append(df)
    
    return weather_df_list