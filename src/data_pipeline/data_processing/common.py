#%%
import pandas as pd
import logging
#%%
def create_exploratory_dataset(production_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """Renvoie le dataset final composé des données de production solaire (production_data) 
    et des données météorologiques (weather_data). Quelques tests simples sont effectués (null values, merge)

    Args:
        production_data (pd.DataFrame): données de production solaire
        weather_data (pd.DataFrame): données météorologiques

    Returns:
        exploratory_df (pd.DataFrame): dataframe exploratoire en vue du merging
    """
    
    try :
        logging.info("Merging production_data, weather_data")
        exploratory_df = pd.merge(production_data, weather_data, left_index=True, right_index=True)
        is_nan = exploratory_df.isna().sum().sum()
        if is_nan > 0:
            logging.warning(f'Données manquantes existantes : {is_nan/len(exploratory_df)}%. Traitement nécessaire lors de la phase exploration')
    
        return exploratory_df
    
    except Exception as e:
        logging.error(f"Erreur lors du merging des données : {e}")
        raise