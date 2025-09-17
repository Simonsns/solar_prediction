#%% Librairies

import requests
import pandas as pd
from typing import Dict, List
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%%
def fetch_solar_data(url: str, 
					 columns: List[str] | None = None, 
					 params: Dict | None = None,
					 ) -> pd.DataFrame:
	"""Fonction basique de requête à un URL api, qui retourne les données solaire en format dictionnaire

	Args:
		url (str): url de l'api
		columns (List[str]): colonnes du DataFrame final
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

#%%
def fetch_inference_solar_data(url: str,  n_records: int, 
							   limit: int = 96, 
							   columns: List[str] | None = None, 
							   params: Dict | None = None):
    
    """Fonction retournant plusieurs appels api (limit REST API) sous forme DataFrame.

    Args:
        url (str): url de l'api
        n_records (int): Nombre de records final
        limit (int, optional): Nombre de lignes par appel. Defaults to 96.
        columns (List[str] | None, optional): Colonnes du DataFrame final. Defaults to None.
        params (Dict | None, optional): Paramètres de filtrage de l'api. Defaults to None.

    Returns:
        pd.DataFrame : DataFrame concaténant l'ensemble des appels api sous forme d'une série temporelle ordonnée par 
		la date.
    """
    
    records = []
    offset = 0
    local_params = params.copy() if params else {}
    local_params.update({"offset": 0, "limit": limit})

    while offset < n_records:
        
        remaining = n_records - offset
        batch_limit = min(limit, remaining)
        local_params.update({"offset": offset, "limit": batch_limit})
        
        # Fetching
        df = fetch_solar_data(url=url, 
                                columns=columns, 
                                params=local_params)

        records.append(df)
        logging.debug(f"Batch récupéré : {df.shape[0]} lignes (offset={offset})")
        
        # Update params
        offset += limit 

    return pd.concat(records, axis=0, ignore_index=True)
# %%