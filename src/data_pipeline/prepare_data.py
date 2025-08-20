# Ce fichier est lle script orchestrateur permettant de collecter (data_collection), transformer (data_processing)
# et valider (data_validation) les résultats
# author : Simon Senegas
# 2025/05/27

# Import libraries
#%%
# Librairies
import pandas as pd
import geopandas as gpd
from dotenv import find_dotenv, load_dotenv
import logging
import os
from typing import List

# Import ETL
from data_collection.weather.fetching_data import fetch_all_hourly_weather_runs
from data_processing.weather.preprocessing import (separate_central_scenario, 
                           set_time_index_drop_date_columns,
                           compute_variable_dispersion, 
                           concatenate_weather_data)
from data_processing.solar.preprocessing import prepare_production_data
from data_processing.common import create_exploratory_dataset
#%%
def setup_logging() -> None:

    """Configure les logs (basique)"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_pipeline.log', encoding="latin-1"),
            logging.StreamHandler()
        ]
    )

def validate_inputs(coordinates_path: str, variables: List[str], 
                   start_date: str, end_date: str) -> None:
    """Valide les paramètres d'entrée de la pipeline"""
    
    # Validation fichier coordonnées
    if not os.path.exists(coordinates_path):
        raise FileNotFoundError(f"Fichier coordonnées introuvable : {coordinates_path}")
    
    # Validation variables
    if not variables:
        raise ValueError("La liste des variables ne peut pas être vide")
    
    # Validation dates
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            raise ValueError("La date de début doit être antérieure à la date de fin")

    except Exception as e:
        raise ValueError(f"Format de date invalide : {e}")
#%% 
def prepare_training_data(weather_url: str, production_link: str, coordinates_path: str, code_region: int,
                 variables: List[str], start_date: str, end_date: str, time_agregation: str) -> pd.DataFrame:
    
    """Fonction qui retourne le dataset exploratoire en vue du training, divisé en deux pipelines: 
    - weather : appel api -> data processing
    - production : appel api ou données locale -> data processing
    -> concaténation des deux dataframes de sorties

    Args:
        weather_url (str): URL de l'api météo
        production_link (str): URL ou lien des données de production
        coordinates_path (str): coordonnées des points de mesures météorologiques
        variables (list): liste des variables météorologiques relevées aux différents points
        start_date (str): date de début de mesure (format YYYY-MM-DD)
        end_date (str): date de fin de mesure (format YYYY-MM-DD)
        time_agregation (str): mesure d'agrégation temporelle

    Returns:
        exploratory_df (pd.DataFrame) : DataFrame exploratoire en vue de la phase de training
    """
    #Informations de base
    logging.info("=" * 60)
    logging.info(["[INIT] Début de la pipeline data"])
    logging.info("=" * 60)
    logging.info(f"Période d'analyse {start_date} → {end_date}")
    logging.info(f'Variables météorologiques mesurées (sans traitement) : {len(variables)}')
    logging.info(f"Agrégation temporelle : {time_agregation}")
    
    # Validation des paramètres de la pipeline
    logging.info("PHASE 1/4 - Validation des paramètres")
    validate_inputs(coordinates_path, variables, start_date, end_date)
    logging.info("✓ : Validation réussie")

    # Gestion des données spatiales
    logging.info("Phase 2/4 - Chargement des coordonnées de mesure météorologiques")
    coordinates = gpd.read_file(coordinates_path)
    logging.info(f'✓ : {len(coordinates)} coordonnées de mesures météorologiques chargées dont {len(coordinates)-1} départementales')

    # Weather pipeline
    logging.info("Phase 3/4 - Début du process données météorologiques")
    df_weather_list = fetch_all_hourly_weather_runs(weather_url, start_date, end_date, variables, coordinates)
    logging.info(f"✓ : Données météo récupérées pour {len(df_weather_list)} scenarios")

    df_cweather, df_other = separate_central_scenario(df_weather_list)
    df_alternative_weather = set_time_index_drop_date_columns(df_other)
    df_dispersion = compute_variable_dispersion(df_alternative_weather, variables)
    df_central_weather = concatenate_weather_data(df_cweather, df_dispersion)

    logging.info(f"✓ Pipeline météo terminée - Shape: {df_central_weather.shape}")

    # Production pipeline
    logging.info("Phase 4/4 - Pipeline production")
    df_production = pd.read_csv(production_link, sep=";", low_memory=False)
    df_prod = prepare_production_data(df_production, code_region, time_agregation)
    logging.info(f"✓ Pipeline production terminée - Shape: {df_prod.shape}")

    #Concatenation
    logging.info("Fusion des datasets en cours")
    exploratory_df = create_exploratory_dataset(df_prod, df_central_weather)
    
    logging.info("=" * 60)
    logging.info("[END] Pipeline data terminée avec succès")
    logging.info("=" * 60)
    logging.info(f"Shape dataset final : {exploratory_df.shape}")
    logging.info(f"Période couverte : {exploratory_df.index.min()} → {exploratory_df.index.max()}")
    logging.info(f"Variables disponibles (dont variable cible) : {list(exploratory_df.columns)}")

    return exploratory_df
#%% Test training data
if __name__ == "__main__":
    
    setup_logging()

    # Trouver les paramètres environnements
    path = find_dotenv()
    load_dotenv(path)
    
    #paramètres env
    coord_path = str(os.getenv("coordinates_path"))
    weather_url = str(os.getenv("API_WEATHER_KEY"))
    production_link = str(os.getenv("production_data_link"))

    #Paramètres utilisateurs
    variables = ["temperature_2m", "sunshine_duration", "is_day", "relative_humidity_2m", 
                            "precipitation", "surface_pressure", "cloud_cover","wind_speed_10m", 
                            "wind_direction_10m", "direct_radiation", "diffuse_radiation", 
                            "direct_normal_irradiance", "shortwave_radiation", "global_tilted_irradiance", 
                            "terrestrial_radiation", "apparent_temperature"]
    start_date = "2023-02-01"
    end_date = "2025-05-16"
    time_agregation = "1h"
    code_region = 76

    exploratory_df = prepare_training_data(weather_url, production_link, coord_path, 
                                  code_region, variables, start_date, end_date, 
                                  time_agregation)
    
    output_path = str(os.getenv("save_link"))
    exploratory_df.to_csv(output_path, sep=";")
    logging.info(f"✓ : Dataset sauvegardé et prêt à l'emploi")
#%%