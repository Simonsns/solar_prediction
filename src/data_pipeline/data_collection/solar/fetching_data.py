#%% Librairies
import requests
import pandas as pd
from retry_requests import retry
import json
import logging
import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#%%
def fetch_solar_production_data(url) -> pd.DataFrame:
	"""Fonction basique de requête à un URL api, qui retourne les données en format dictionnaire

	Args:
		url (str): url de l'api
		params (dict): paramètres de filtrage sql de l'api

	Returns:
		data (dict): données sous forme de dictionnaire
	"""
	now = datetime.datetime.today()
	day = now.strftime("%Y-%m-%d")
	solar_params = {"select":"code_insee_region, date, heure, solaire", 
					"where" : f"code_insee_region=76 AND date='{day}'",
					"order_by":"heure", 
					"limit": "96"}
	try:
		response = requests.get(url, params=solar_params)
		response.raise_for_status()
		json_data = response.json()
		data = pd.DataFrame(json_data.get("results"))
		logging.info("Production solaire en temps réel téléchargée avec succès")
		return data 
	
	except requests.RequestException as e:
		logging.error(f"Erreur lors de la requête API: {e}")
		raise