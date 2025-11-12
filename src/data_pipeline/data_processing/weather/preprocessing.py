import pandas as pd
import logging
#%%
def separate_central_scenario(weather_df_list: list):
    
    """Separe le scénario central (dernier élément de la liste) des autres dataframes"""
    
    df_central = weather_df_list[-1] # DataFrame scénario central
    df_others = weather_df_list[:-1] # Autres dataframes (ici points de mesures départementaux)
    df_other = pd.concat(df_others, axis=1)
    
    return df_central, df_other
#%%
def set_time_index_drop_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dates superflues provenant de l'appel à l'api"""
    
    date_col_index = [col for col in df.columns if col.startswith("date")][0]
    df = df.set_index(date_col_index)
    df = df.drop(columns=[col for col in df.columns if col.startswith("date")])

    return df
#%%
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

#%%
def concatenate_weather_data(df_weather: pd.DataFrame, df_dispersion: pd.DataFrame) -> pd.DataFrame:
    
    """Concatène le scénario météo central (df_weather), la dispersion des variables météo (df_dispersion),
    les données de production solaire (df_production), en gérant les agrégations temporelles"""
    
    date_col_index = [col for col in df_weather.columns if col.startswith("date")][0]
    df_central = df_weather.set_index(date_col_index)
    df_central = pd.merge(df_central, df_dispersion, left_index=True, right_index=True)

    return df_central

#%%