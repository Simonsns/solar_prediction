import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def separate_central_scenario(weather_df_list: list):
    
    """Separe le scénario central (dernier élément de la liste) des autres dataframes"""
    
    df_central = weather_df_list[-1] # DataFrame scénario central
    df_others = weather_df_list[:-1] # Autres dataframes (ici points de mesures départementaux)
    df_other = pd.concat(df_others, axis=1)
    
    return df_central, df_other

def set_time_index_drop_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dates superflues provenant de l'appel à l'api"""
    
    date_col_index = [col for col in df.columns if col.startswith("date")][0]
    df = df.set_index(date_col_index)
    df = df.drop(columns=[col for col in df.columns if col.startswith("date")])

    return df

def compute_variable_dispersion(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Compute les écarts min-max et la dispersion des différents points de mesure sur 
    les différentes variables listées"""

    df_result = pd.DataFrame(index=df.index)
    
    for var in variables : 
        logging.info(f"{var} en cours de traitement")
        col_filter = [col for col in df.columns if col.startswith(var)]
        data = df[col_filter]
        df_result[f"{var}_delta_minmax"] = (data.max(axis=1) - data.min(axis=1)).round(4)
        df_result[f"{var}_std"] = data.std(axis=1).round(4)

    logging.info("Traitement des variables météorologiques terminé")

    return df_result

def prepare_production_data(production_data: pd.DataFrame, code_region: int, time_agregation: str) -> pd.DataFrame:
    
    """Préparation des données de production : 
    - Filtrage sur la région étudiée
    - Mise en place d'un index temporel avec agrégation choisie
    Retourne ensuite un DataFrame (df_prod) composé des attributs étudiés.

    Args:
        production_data (pd.DataFrame): Données de production solaire
        code_region (int): Code de la région étudiée
        time_agregation (str): mesure d'agrégation temporelle

    Returns:
        df prod (pd.DataFrame): Données de production solaire
    """
    
    df_prod = production_data[production_data["Code INSEE région"]==code_region][["Date - Heure", "Solaire (MW)"]]
    df_prod["Date - Heure"] = pd.to_datetime(df_prod["Date - Heure"], utc=True).dt.tz_convert('Europe/Paris') # Manipulation manuelle pour controler le pipe entier
    
    df_prod = df_prod.sort_values('Date - Heure') 
    df_prod = df_prod.set_index("Date - Heure")
    df_prod = df_prod.resample(time_agregation).mean() # Agrégation 

    return df_prod

def concatenate_weather_data(df_weather: pd.DataFrame, df_dispersion: pd.DataFrame) -> pd.DataFrame:
    
    """Concatène le scénario météo central (df_weather), la dispersion des variables météo (df_dispersion),
    les données de production solaire (df_production), en gérant les agrégations temporelles"""
    
    date_col_index = [col for col in df_weather.columns if col.startswith("date")][0]
    df_central = df_weather.set_index(date_col_index)
    df_central = pd.merge(df_central, df_dispersion, left_index=True, right_index=True)

    return df_central

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