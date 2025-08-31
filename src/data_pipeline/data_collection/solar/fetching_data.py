#%% Librairies
import requests
import pandas as pd
from typing import Dict, List
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#%%
def fetch_solar_data(url: str, columns: List[str] | None = None, params: Dict | None = None) -> pd.DataFrame:
	"""Fonction basique de requête à un URL api, qui retourne les données solaire en format dictionnaire

	Args:
		url (str): url de l'api
		params (dict): paramètres de filtrage sql de l'api

	Returns:
		data (dict): données sous forme de dictionnaire
	"""
	try:
		response = requests.get(url, params=params) 
		response.raise_for_status()
		json_data = response.json()
		data = pd.DataFrame(data=(json_data.get("results", [])), columns=columns)

		logging.info("Data solaire téléchargée avec succès")
		return data
		
	except requests.RequestException as e:
		logging.error(f"Erreur lors de la requête API: {e}")
		raise