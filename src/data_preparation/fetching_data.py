
import requests
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import json

def fetching_data(url, params) -> dict:
    """Fonction basique de requête à un URL api, qui retourne les données en format dictionnaire

    Args:
        url (str): url de l'api
        params (dict): paramètres de filtrage sql de l'api

    Returns:
        data (dict): données sous forme de dictionnaire
    """
    response = requests.get(url, params=params)
    json_file = response.json()
    data = json.loads(json_file)
    
    return data


def fetching_weather_historical_data(url: str, start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
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
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	params = {
		"latitude": [lat],
		"longitude": [lon],
		"start_date": f'{start_date}',
		"end_date": f'{end_date}',
		"hourly": ["temperature_2m", "sunshine_duration", "is_day", "relative_humidity_2m", "precipitation", "surface_pressure", "cloud_cover", "visibility", "temperature_80m", "wind_speed_10m", "wind_speed_80m", "wind_direction_10m", "wind_direction_80m", "temperature_120m", "uv_index", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation", "global_tilted_irradiance", "terrestrial_radiation", "apparent_temperature"]
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_sunshine_duration = hourly.Variables(1).ValuesAsNumpy()
	hourly_is_day = hourly.Variables(2).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
	hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
	hourly_visibility = hourly.Variables(7).ValuesAsNumpy()
	hourly_temperature_80m = hourly.Variables(8).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(9).ValuesAsNumpy()
	hourly_wind_speed_80m = hourly.Variables(10).ValuesAsNumpy()
	hourly_wind_direction_10m = hourly.Variables(11).ValuesAsNumpy()
	hourly_wind_direction_80m = hourly.Variables(12).ValuesAsNumpy()
	hourly_temperature_120m = hourly.Variables(13).ValuesAsNumpy()
	hourly_uv_index = hourly.Variables(14).ValuesAsNumpy()
	hourly_direct_radiation = hourly.Variables(15).ValuesAsNumpy()
	hourly_diffuse_radiation = hourly.Variables(16).ValuesAsNumpy()
	hourly_direct_normal_irradiance = hourly.Variables(17).ValuesAsNumpy()
	hourly_shortwave_radiation = hourly.Variables(18).ValuesAsNumpy()
	hourly_global_tilted_irradiance = hourly.Variables(19).ValuesAsNumpy()
	hourly_terrestrial_radiation = hourly.Variables(20).ValuesAsNumpy()
	hourly_apparent_temperature = hourly.Variables(21).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["sunshine_duration"] = hourly_sunshine_duration
	hourly_data["is_day"] = hourly_is_day
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["precipitation"] = hourly_precipitation
	hourly_data["surface_pressure"] = hourly_surface_pressure
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["visibility"] = hourly_visibility
	hourly_data["temperature_80m"] = hourly_temperature_80m
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
	hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
	hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
	hourly_data["wind_direction_80m"] = hourly_wind_direction_80m
	hourly_data["temperature_120m"] = hourly_temperature_120m
	hourly_data["uv_index"] = hourly_uv_index
	hourly_data["direct_radiation"] = hourly_direct_radiation
	hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
	hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
	hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
	hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance
	hourly_data["terrestrial_radiation"] = hourly_terrestrial_radiation
	hourly_data["apparent_temperature"] = hourly_apparent_temperature

	hourly_weather_df = pd.DataFrame(data = hourly_data)

	return hourly_weather_df