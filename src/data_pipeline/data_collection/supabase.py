# Fichier pour toutes les fonctions utilisées avec supabase
# author : simon.senegas
# date : 12/11/2025

import pandas as pd
from supabase import Client, create_client
import geopandas as gpd
import logging
import numpy as np

def extract_coordinates_from_supabase(supabase_url: str, supabase_key: str, table_name: str):
    """
    Récupère les coordonnées des barycentres de capacité solaire depuis Supabase 
    et la transforme en GeoDataFrame.
    """
    # Initialisation du client
    supabase: Client = create_client(supabase_url=supabase_url, 
                                     supabase_key=supabase_key)
    try:
        response = supabase.table(table_name).select("id", "geometry").execute()
        df = pd.DataFrame(response.data)
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry="geometry")

    except Exception as e :
        logging.warning("Erreur lors de l'import des coordonnées", e)
        raise

    return gdf
def refresh_supabase_inference_table(inference_dataset: pd.DataFrame,
                                supabase_url: str,
                                supabase_key: str,
                                table_name: str):
    """Met à jour (refresh) une table d'inférence complète dans Supabase.

    L'opération s'effectue en trois étapes :
    1. Nettoyage et conversion du DataFrame (NaN -> None, Timestamp -> str ISO).
    2. Suppression complète des anciennes données via un appel de fonction RPC (TRUNCATE).
    3. Insertion du nouveau jeu de données dans la table.

    Args:
            inference_dataset (pd.DataFrame): Le DataFrame contenant les nouvelles données
            d'inférence. Les NaN sont convertis en None et les Timestamps en format ISO.
            supabase_url (str): L'URL de votre projet Supabase.
            supabase_key (str): La clé d'accès (généralement la clé anon) pour l'API Supabase.
            table_name (str): Le nom de la table Supabase à rafraîchir (ex: 'INFERENCE_TABLE').

    Returns:
            None: La fonction ne retourne rien mais effectue une opération d'écriture
            dans la base de données et logue le résultat.
    """
    supabase: Client = create_client(supabase_url=supabase_url,
                                    supabase_key=supabase_key)

    # Type Check
    inference_dataset = inference_dataset.convert_dtypes()
    inference_dataset = inference_dataset.reset_index()
    inference_dataset["solaire"] = inference_dataset["solaire"].replace({np.nan: None})
    for col in inference_dataset.select_dtypes(include=['datetime', 'datetimetz']).columns:
            inference_dataset[col] = inference_dataset[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Initialisation
    data_dict = inference_dataset.to_dict(orient="records")

    try:
        delete_response = supabase.rpc('refresh_inference_table', {}).execute()
        logging.info("Suppression du précédent dataset d'inférence", delete_response)
        insert_response = supabase.table(table_name).insert(data_dict).execute()
        logging.info("Dataset d'inférence stocké sur Supabase", insert_response)
    
    except Exception as e:
        logging.warning("Dataset d'inférence non stocké sur Supabase", e)
        raise