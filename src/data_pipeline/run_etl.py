"""
Pipeline ETL pour la prédiction de production solaire
author : Simon.sngs
date : 15/11/2025
"""
#%%
# Utilitaires
import os
import logging
from typing import Dict, Any
from pathlib import Path

# Gestion de BDD
import pandas as pd
import geopandas as gpd

# Modules 
from src import config
from src.data_pipeline.data_processing.solar.preprocessing import prepare_production_data
from src.data_pipeline.data_collection.solar.fetching_data import (
    fetch_inference_solar_data,
    fetch_solar_data
)
from src.data_pipeline.data_processing.installed_solar_capacity.preprocessing import (
    inference_installed_power_sum
)
from src.data_pipeline.data_collection.weather.fetching_data import (
    fetch_forecast_weather,
    fetch_historical_weather
)
from src.data_pipeline.data_processing.feature_engineering import (
    full_raw_inference_dataset,
    transform_pipeline
)
from src.data_pipeline.data_collection.supabase import (
    extract_coordinates_from_supabase,
    refresh_supabase_inference_table
)

# Setup config
# Créer le dossier logs s'il n'existe pas
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "data_pipeline.log"

# Configuration avec fichier et console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # Affiche aussi dans la console
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("supabase").setLevel(logging.ERROR)


def load_environment_variables() -> Dict[str, str]:
    """
    Charge les variables d'environnement nécessaires.
    
    Returns:
        Dict contenant les URLs et clés API
    
    Raises:
        EnvironmentError: Si une variable requise est manquante
    """
    required_vars = [
        "API_SOLAR_KEY",
        "API_CAPACITY_KEY",
        "API_WEATHER_KEY",
        "API_FORECAST_WEATHER_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY"
    ]
    
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = str(os.getenv(var))
        if not value:
            missing_vars.append(var)
        env_vars[var] = value
    
    if missing_vars:
        raise EnvironmentError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}"
        )
    
    return env_vars


def extract_installed_capacity(cap_production_url: str, region_code: int):
    """
    Extract: Récupère la capacité installée pour la région.
    
    Args:
        cap_production_url: URL de l'API de capacité
        region_code: Code de la région
    
    Returns:
        Somme de la capacité installée
    """
    logger.info("Extraction de la capacité installée...")
    installed_power = fetch_solar_data(url=cap_production_url)
    sum_capacity = inference_installed_power_sum(df_installed_capacity=installed_power, 
                                                 region_code=region_code)
    
    logger.info(f"Capacité installée récupérée: {sum_capacity} MW")
    
    return sum_capacity


def extract_coordinates(supabase_url: str, supabase_key: str, 
                       coord_table: str) -> gpd.GeoDataFrame:
    """
    Extract: Récupère les coordonnées géographiques depuis Supabase.
    
    Args:
        supabase_url: URL Supabase
        supabase_key: Clé API Supabase
        coord_table: Nom de la table des coordonnées
    
    Returns:
        DataFrame contenant les coordonnées
    """
    logger.info("Extraction des coordonnées de Supabase...")
    coordinates = extract_coordinates_from_supabase(
        supabase_key=supabase_key,
        supabase_url=supabase_url,
        table_name=coord_table
    )
    logger.info(f"{len(coordinates)} coordonnées/scénarios extraites")
    
    return coordinates


def extract_production_data(solar_inference_api: str, n_records: int,
                           params: Dict[str, Any], region_code: int,
                           time_agregation: str) -> pd.DataFrame:
    """
    Extract: Récupère et prépare les données de production solaire.
    
    Args:
        solar_inference_api: URL de l'API solaire
        n_records: Nombre d'enregistrements à récupérer
        params: Paramètres de l'API
        region_code: Code de la région
        time_agregation: Niveau d'agrégation temporelle
    
    Returns:
        DataFrame des données de production préparées
    """
    logger.info("Extraction des données de production solaire...")
    inference_solar_data = fetch_inference_solar_data(
        url=solar_inference_api,
        n_records=n_records,
        params=params
    )
    
    production_data = prepare_production_data(
        inference_solar_data,
        region_code,
        time_agregation,
        "inference"
    )

    logger.info(f"{len(production_data)} enregistrements de production extraits")
    
    return production_data


def load_data(inference_data: pd.DataFrame, supabase_url: str,
             supabase_key: str, inference_table: str) -> None:
    """
    Load: Charge les données transformées dans Supabase.
    
    Args:
        inference_data: Données d'inférence transformées
        supabase_url: URL Supabase
        supabase_key: Clé API Supabase
        inference_table: Nom de la table d'inférence
    """
    logger.info("Chargement des données dans Supabase...")
    refresh_supabase_inference_table(
        inference_dataset=inference_data,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=inference_table
    )
    logger.info(f"{len(inference_data)} enregistrements chargés avec succès")


def run_etl() -> None: #TODO Faire une configuration propre avec Pydantic
    """
    Exécute le pipeline ETL complet pour la prédiction de production solaire.
    
    Pipeline:
        1. Extract: Capacité installée, coordonnées, production, météo
        2. Transform: Feature engineering et normalisation
        3. Load: Chargement dans Supabase
    
    Raises:
        Exception: En cas d'erreur durant le pipeline
    """
    try:
        logger.info("=" * 80)
        logger.info("DÉMARRAGE DU PIPELINE ETL")
        logger.info("=" * 80)
        
        # Chargement des variables d'environnement
        env_vars = load_environment_variables()
        
        # Chargement des paramètres de configuration
        time_agregation = config.TIME_AGREGATION
        n_records = config.N_RECORDS
        len_prev = config.LEN_PREV
        params = config.RTE_DEFAULT_PARAMS
        region_code = config.REGION_CODE
        
        # Feature engineering
        central_scenario = config.CENTRAL_SCENARIO
        lagged_feature_list = config.LAGGED_FEATURE_LIST
        timeframe_dict = config.TIMEFRAME_DICT
        lag_list = config.LAG_LIST
        window_list = config.WINDOW_LIST
        
        # Paramètres API
        variables = config.API_WEATHER_VARIABLES
        coord_table = config.COORD_TABLE
        inference_table = config.INFERENCE_TABLE
        
        # ============================================================
        # EXTRACT
        # ============================================================
        logger.info("EXTRACTION DES DONNÉES")
        logger.info("-" * 80)
        
        # Capacité installée
        sum_capacity = extract_installed_capacity(
            cap_production_url=env_vars["API_CAPACITY_KEY"],
            region_code=region_code
        )
        
        # Coordonnées
        coordinates = extract_coordinates(
            supabase_url=env_vars["SUPABASE_URL"],
            supabase_key=env_vars["SUPABASE_KEY"],
            coord_table=coord_table
        )
        
        # Données de production
        production_data = extract_production_data(
            solar_inference_api=env_vars["API_SOLAR_KEY"],
            n_records=n_records,
            params=params,
            region_code=region_code,
            time_agregation=time_agregation
        )
        
        # Données météorologiques
        logger.info("Extraction des données météorologiques historiques")

        historical_weather = fetch_historical_weather(production_data=production_data,
                                              variables=variables,
                                              coordinates=coordinates,
                                              weather_url=env_vars["API_WEATHER_KEY"])
        
        logger.info("Extraction des données météorologiques prévisionnelles")

        forecast_weather = fetch_forecast_weather(production_data=production_data,
                                                variables=variables,
                                                len_prev = len_prev,
                                                coordinates=coordinates,
                                                forecast_weather_url=env_vars["API_FORECAST_WEATHER_KEY"])
        
        # ============================================================
        # TRANSFORM
        # ============================================================
        logger.info("TRANSFORMATION DES DONNÉES")
        logger.info("-" * 80)
        
        # Concaténation météo/solaire et noramlisation de la production
        raw_inference_data = full_raw_inference_dataset(production_data=production_data, 
                                historical_weather=historical_weather,
                                forecast_weather=forecast_weather, 
                                sum_capacity=sum_capacity)

        # Transform
        inference_data = transform_pipeline(raw_inference_data=raw_inference_data,
                                            timeframe_dict=timeframe_dict,
                                            lag_list=lag_list,
                                            window_list=window_list,
                                            lagged_feature_list=lagged_feature_list,
                                            central_scenario=central_scenario)
        
        # ============================================================
        # LOAD
        # ============================================================
        logger.info("CHARGEMENT DES DONNÉES")
        logger.info("-" * 80)
        
        load_data(
            inference_data=inference_data,
            supabase_url=env_vars["SUPABASE_URL"],
            supabase_key=env_vars["SUPABASE_KEY"],
            inference_table=inference_table
        )
        
        logger.info("=" * 80)
        logger.info("PIPELINE ETL TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"ERREUR DANS LE PIPELINE ETL: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_etl()
# %%
