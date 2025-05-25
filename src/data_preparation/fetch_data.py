
import requests
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(url, params) -> dict:
    """Fonction basique de requête à un URL api, qui retourne les données en format dictionnaire

    Args:
        url (str): url de l'api
        params (dict): paramètres de filtrage sql de l'api

    Returns:
        data (dict): données sous forme de dictionnaire
    """
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return json.loads(response)
    
    except requests.RequestException as e:
        
        logging.error(f"Erreur lors de la requête API: {e}")
        raise

#%%
def fetch_hourly_hist_weather_data(url: str, start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
	
	"""Appel à l'API météo. Retourne en sortie un dataframe avec 22 features météorologiques aux coordonnées données.

	Args:
    	url (str): url de l'api
		start_date (str): Date de début des données
		end_date (str): Date de fin des données
        lat (float): latitude du point météorologique étudié
        lon (float): longitude du point météorologique étudié

	Returns:
		hourly_weather_df (pd.DataFrame): DataFrame avec les variables météorologiques horaire sur la fenêtre temporelle donnée
	"""
	# Initialisation du client OPEN-METEO
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)
	
	# Nom des variables
	variable_names = ["temperature_2m", "sunshine_duration", "is_day", "relative_humidity_2m", 
                        "precipitation", "surface_pressure", "cloud_cover", "visibility", 
                        "temperature_80m", "wind_speed_10m", "wind_speed_80m", "wind_direction_10m", 
                        "wind_direction_80m", "temperature_120m", "uv_index", "direct_radiation", 
                        "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation", 
                        "global_tilted_irradiance", "terrestrial_radiation", "apparent_temperature"]
	
	# Toutes les variables sont mentionnées ici 
	params = {
		"latitude": [lat],
		"longitude": [lon],
		"start_date": f'{start_date}',
		"end_date": f'{end_date}',
		"hourly": variable_names,
	}

	try:
		responses = openmeteo.weather_api(url, params=params)
		response = responses[0]

		logging.info(f"Coordonnées : {response.Latitude()}°N, {response.Longitude()}°E")
		logging.info(f"Altitude : {response.Elevation()} m")
		logging.info(f"Fuseau horaire : {response.Timezone()} ({response.TimezoneAbbreviation()})")

		hourly = response.Hourly()
		time_range = pd.date_range(
			start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
			end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
			freq=pd.Timedelta(seconds=hourly.Interval()),
			inclusive="left"
		)

		hourly_data = {"date": time_range}

		for i, var in enumerate(variable_names):
			hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

		return pd.DataFrame(data=hourly_data)

	except Exception as e:
		logging.error(f"Erreur lors de la récupération des données météo : {e}")
		raise