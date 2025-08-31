#%%
import pandas as pd
import logging
import os
#%%
def normalize_solar_data(df_installed_power: pd.DataFrame, df_solar_weather: pd.DataFrame) -> pd.DataFrame:
    """Retourne le dataset d'entrainement avec la variable cible (solaire) normalisée"""
    
    # Initialisation
    normalized_solar_mw = pd.Series((df_solar_weather["Solaire (MW)"]/df_installed_power["chronique_capacity"]), 
                                    name="normalized_solar_mw")
    normalized_solar_mw = normalized_solar_mw.dropna()
    
    # Gestion des données manquantes
    if len(normalized_solar_mw) == df_solar_weather.shape[0]:
        df_solar_weather["normalized_solar_mw"] = normalized_solar_mw
        logging.info(f'DataFrame normalisé avec succès par la capacité de production solaire régionale')
    else:
        delta = (len(normalized_solar_mw) - df_solar_weather.shape[0])
        logging.info(f'{delta} données manquantes lors de la normalisation')
        df_solar_weather = df_solar_weather.merge(normalized_solar_mw, left_index=True, right_index=True, how="left")
        df_solar_weather["normalized_solar_mw"] = df_solar_weather["normalized_solar_mw"].interpolate(method="linear") 
        logging.info('DataFrame normalisé et interpolé avec succès par la capacité de production solaire régionale')

    return df_solar_weather
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
        logging.info("Merge solar_data & weather_data")
        exploratory_df = pd.merge(production_data, weather_data, left_index=True, right_index=True)
        is_nan = exploratory_df.isna().sum().sum()
        if is_nan > 0:
            logging.warning(f'Données manquantes existantes : {is_nan/len(exploratory_df)}%. Traitement nécessaire lors de la phase exploration')
    
        return exploratory_df
    
    except Exception as e:
        logging.error(f"Erreur lors du merging des données : {e}")
        raise