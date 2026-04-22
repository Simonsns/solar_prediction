#%%
import pandas as pd
#%%
def prepare_production_data(production_data: pd.DataFrame, 
                        code_region: int, 
                        time_agregation: str,
                        data_type: str) -> pd.DataFrame:
    
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
    
    if data_type=="training":
        df_prod = production_data[production_data["Code INSEE région"]==code_region][["Date - Heure", "Solaire (MW)"]]
        df_prod.loc[:, "Date - Heure"] = pd.to_datetime(df_prod["Date - Heure"], utc=True).dt.tz_convert('Europe/Paris') # Manipulation manuelle pour controler le pipe entier
        
        df_prod = df_prod.sort_values('Date - Heure') 
        df_prod = df_prod.set_index("Date - Heure")
        df_prod = df_prod.resample(time_agregation).mean() # Agrégation 

    else:
        df_prod = production_data[["date_heure", "solaire"]]
        df_prod.loc[:, "date_heure"] = pd.to_datetime(df_prod["date_heure"], utc=True).dt.tz_convert('Europe/Paris') # Manipulation manuelle pour controler le pipe entier
        
        df_prod = df_prod.sort_values('date_heure') 
        df_prod = df_prod.set_index("date_heure")
        df_prod = df_prod.resample(time_agregation).mean() # Agrégation 

    return df_prod.dropna()