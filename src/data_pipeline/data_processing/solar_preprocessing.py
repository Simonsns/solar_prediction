#%%
import pandas as pd
from typing import Optional

def training_filter(df: pd.DataFrame,
                    code_region: int
                    ) -> pd.DataFrame:

    filtered_df = df.loc[
            df["code_insee_region"] == str(code_region),
            ["date_heure", "solaire"]
        ].copy()
    
    return filtered_df

def prepare_production_data(production_data: pd.DataFrame, 
                        code_region: int, 
                        time_agregation: str,
                        data_type: str,
                        start_training_date: Optional[str] = None,
                        end_training_date: Optional[str] = None
                        ) -> pd.DataFrame:
    
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
    
    # Init
    required_cols = {"date_heure", "solaire"}

    if not required_cols.issubset(production_data.columns):
        raise ValueError(f"Missing columns: {required_cols - set(production_data.columns)}")
    
    if production_data.empty:
        raise ValueError("production_data is empty")
    
    df = production_data.copy()
    
    # Datetime
    df["date_heure"] = (
        pd.to_datetime(df["date_heure"], utc=True)
        .dt.tz_convert("Europe/Paris")
        ) 
    
    # Filtering (environment conditional)
    if data_type.lower()=="training":
        if not start_training_date or not end_training_date:
            raise ValueError("start_training_date and end_training_date are required for training mode.")
        
        # Filtering
        start_ts = pd.to_datetime(start_training_date).tz_localize("Europe/Paris")
        end_ts = pd.to_datetime(end_training_date).tz_localize("Europe/Paris")
        df = df[(df["date_heure"] >= start_ts) & (df["date_heure"] <= end_ts)]
        df_prod = training_filter(df=df, code_region=code_region)

    else:
        df_prod = df[["date_heure", "solaire"]].copy()
    
    # Indexing and sorting
    df_prod = ( df_prod
               .sort_values("date_heure")
               .set_index("date_heure")
               .resample(time_agregation)
               .mean()
    )

    if df_prod["solaire"].isna().all():
        raise ValueError("DataFrame is empty after filtering and agregation")
    
    return df_prod.dropna()

