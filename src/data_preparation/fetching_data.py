#%%
import requests
from tenacity import retry
#%%
def fetching_data(url: str, params: dict):
    """_summary_

    Args:
        url (str): url de l'api
        params (dict): paramètres de filtrage sql de l'api

    Returns:
        json: La réponse en fichier json
    """
    response = requests.get(url, params=params)

    return response.json()
# %%