import requests
import pandas as pd
from retry_requests import retry
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_installed_power(url: str) -> pd.DataFrame:
	"""Retourne la capacité installée nationale téléchargée en format PARQUET sous forme de DataFrame"""
	try:
		response = requests.get(url)
		response.raise_for_status()
		df = pd.read_parquet(BytesIO(response.content))
		return df

	except requests.RequestException as e:
		logging.error(f'Erreur lors de la requête API {e}')
		raise

