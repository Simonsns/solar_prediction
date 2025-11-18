# Fonctions de fetching de la production solaire sur l'API RTE
# author : simon.senegas
# date : 18/11/2025

# Librairies
#API 
import requests
from tenacity import (retry, stop_after_attempt, 
                      wait_exponential, 
                      retry_if_exception_type, 
                      before_sleep_log)

#Gestion de daya
import pandas as pd
from typing import Dict, List
import logging
from io import BytesIO
from json import JSONDecodeError

# Configuration et Gestion des exceptions 
RETRY_LOGGER = logging.getLogger('solar_api_retry')
RETRY_LOGGER.setLevel(logging.WARNING) 

V_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,  #5xx et 429
)

@retry(stop=stop_after_attempt(10),
       wait=wait_exponential(multiplier=2, min=10, max=120),
       retry=retry_if_exception_type(V_EXCEPTIONS),
    before_sleep=before_sleep_log(RETRY_LOGGER, logging.WARNING, exc_info=True)
)
def fetch_solar_data(url: str, 
                     columns: List[str] | None = None, 
                     params: Dict | None = None,
                     ):
    """Fonction basique de requête à un URL api, qui retourne les données solaire en format DataFrame Pandas

    Args:
        url (str): url de l'api
        columns (List[str]): colonnes du DataFrame final
        params (dict): paramètres de filtrage sql de l'api

    Returns:
        data (dict): données sous forme de dictionnaire
    """

    response = requests.get(url, params=params)
    response.raise_for_status()

    try:
        json_data = response.json()
        data = pd.DataFrame(data=(json_data.get("results", [])), columns=columns)
        logging.info("Data solaire téléchargée avec succès au format JSON")

    except (ValueError, JSONDecodeError):
        logging.debug("Réponse non JSON, lecture en Parquet")
        data = pd.read_parquet(BytesIO(response.content), columns=columns)
        logging.info("Data solaire téléchargée avec succès au format Parquet")

    return data

def fetch_inference_solar_data(url: str,  n_records: int, 
							   limit: int = 96, 
							   columns: List[str] | None = None, 
							   params: Dict | None = None) -> pd.DataFrame:
    
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
        
        try:
        # Fetching
            df = fetch_solar_data(url=url, 
                                    columns=columns, 
                                    params=local_params)

            records.append(df)
            logging.debug(f"Batch récupéré : {df.shape[0]} lignes (offset={offset})")
        
        except requests.RequestException as e:
            logging.error(f"échec de l'extract API RTE pour le batch (offset={offset}). Motif: {type(e).__name__} - {e}")
        
        # Update params
        offset += limit 
    
    if not records: 
        logging.warning("Aucune donnée de production solaire n'a pu être récupérée de l'API RTE")
        return pd.DataFrame()

    return pd.concat(records, axis=0, ignore_index=True)
