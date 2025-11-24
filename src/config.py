from datetime import datetime, timedelta

# --- PARAMÈTRES GÉNÉRAUX D'EXÉCUTION ---
TIME_AGREGATION = "1h"
REGION_CODE = 76
N_HOURS_TO_FETCH = 99

# Variables temporelles
DATE_START_CALC = datetime.now() - timedelta(hours=N_HOURS_TO_FETCH) 
DATE_START_STR = DATE_START_CALC.strftime("%Y-%m-%d %H:%M:%S")

# Variables API RTE Production solaire en temps réel
# Le script va chercher 99 heures de données agrégées (99 * 4 = 396 points)
N_RECORDS = N_HOURS_TO_FETCH * 4 
RTE_LIMIT = 96 # Limite de résultats par requête (à ajuster selon votre API)
RTE_DEFAULT_PARAMS = {
    "select": "code_insee_region, date, heure, date_heure, solaire",
    "order_by": "date_heure",
    "where": f"code_insee_region='{REGION_CODE}' AND date_heure >= '{DATE_START_STR}'"
}
LEN_PREV = 24 # Longueur de la fenêtre de prédiction

############# Paramètres API Open-météo ##############

API_WEATHER_VARIABLES = [
    "temperature_2m", 
    "relative_humidity_2m", 
    "precipitation", 
    "surface_pressure", 
    "cloud_cover",
    "wind_speed_10m", 
    "wind_direction_10m", 
    "global_tilted_irradiance"
]

########### Paramètres Feature engineering ############

CENTRAL_SCENARIO = 13 # Scénario barycentrique de toute la capacité solaire régionale
LAG_LIST = [1, 6, 24, 48] # Périodes de décalage des séries temporelles
WINDOW_LIST = [6, 24, 48]
LAGGED_FEATURE_LIST = [
    'solaire',
    'global_tilted_irradiance', 
    'temperature_2m', 
    'wind_speed_10m'
]

TIMEFRAME_DICT = {
    "month": 12, 
    "hour": 24
}

################### Tables supabase ###########################

COORD_TABLE = "COORD_TABLE"
INFERENCE_TABLE = "INFERENCE_TABLE"