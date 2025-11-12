#%%
import pandas as pd
import logging
import numpy as np
# %%
def solar_puissance_filter(df_installed_puissance: pd.DataFrame, region_code: int) -> pd.DataFrame: 
    """Retourne le dataset de la puissance solaire installée (après appel api) sur la région de code region_code.

    Args:
        df_installed_puissance (pd.DataFrame): Dataset issu du registre des infrastructures de production d'énergie (API data.gouv)
        region_code (int): Code INSEE de la région

    Returns:
        df_solar_puissance (pd.DataFrame) : Dataset puissance solaire installée sur la région code_region
    """
    # Solar filter
    reg_capacity = df_installed_puissance.groupby(["coderegion", "datemiseenservice", "filiere"])["puismaxinstallee"].sum()
    reg_capacity = reg_capacity.reset_index()
    df_solar_puissance = reg_capacity[reg_capacity["filiere"]=="Solaire"].drop(columns="filiere")
    
    # Spatial filter
    df_solar_puissance["coderegion"] = df_solar_puissance["coderegion"].astype(int)
    df_solar_puissance = df_solar_puissance[df_solar_puissance["coderegion"] == region_code]

    # Time conversion
    df_solar_puissance.loc[:, "datemiseenservice"] = pd.to_datetime(df_solar_puissance["datemiseenservice"], format="%d/%m/%Y")

    return df_solar_puissance

#%%
def delete_temporal_outlier(df_solar_puissance: pd.DataFrame) -> pd.DataFrame:
    """Supprime les outliers temporels basés sur une date de mise en service anormalement ancienne.

    Args:
        df_solar_puissance (pd.DataFrame): DataFrame contenant les données de production solaire, 
        avec une colonne 'datemiseenservice' et 'puismaxinstallee'.

    Returns:
        pd.DataFrame: DataFrame nettoyé sans les outliers temporels, sauf s'ils sont significatifs.
    """
    # Filtrage des outliers temporels
    filter = (df_solar_puissance["datemiseenservice"] < pd.Timestamp("1990-01-01"))
    ancient_solar_production = df_solar_puissance[filter]
    df_puiss = df_solar_puissance[~filter]

    # Total des puissances outliers et correctes
    total_puiss = df_puiss["puismaxinstallee"].sum()
    outlier_puiss = ancient_solar_production["puismaxinstallee"].sum()
    pourcentage_outliers = outlier_puiss / total_puiss * 100
    

    if pourcentage_outliers >= 0.01:
        
        df_solar_puissance.loc[filter, "datemiseenservice"] = df_puiss["datemiseenservice"].iloc[0]
        logging.info(f"{pourcentage_outliers:.2f}% du total détectés comme outliers temporels -> outliers conservés et initialisés à la première date du dataframe de départ")
        
        return df_solar_puissance
    
    else:
        logging.info(f"{pourcentage_outliers:.2f}% du total détectés comme outliers temporels -> outliers supprimés car négligeables")

    return df_puiss
#%%
def cumulative_solar_puissance(df_solar_puissance: pd.DataFrame, start_date: str, end_date: str, resample_method: str = 'W'):
    """Retourne l'intervalle pertinent df_total_capacity considéré dans l'appel d'api de la production solaire RTE.

    Args:
        df_solar_puissance (pd.DataFrame): Dataset filtré des infrastructures de production solaire
        start_date (str): date initiale de l'intervalle 
        end_date (str): date finale de l'intervalle
        resample_method (str, optional): Argument donnant l'agrégation temporelle (W pour week, H pour heure...). Defaults to 'W'.

    Returns:
        df_total_capacity (pd.DataFrame) : Dataset filtré sur l'intervalle (start_date, end_date), munie de la courbe cumulée de puissance associée.
    """
    # Separation de la droite et de la gauche
    df_capacity_r = df_solar_puissance[df_solar_puissance["datemiseenservice"] >= pd.Timestamp(start_date)]
    df_capacity_r = df_capacity_r.set_index('datemiseenservice')
    df_capacity_l = df_solar_puissance[df_solar_puissance["datemiseenservice"] < pd.Timestamp(start_date)]
   
    # Calcul de la baseline
    baseline_capacity = df_capacity_l["puismaxinstallee"].sum()

    # Initialisation de la somme cumulée
    df_capacity_r.loc[:, "chronique_capacity"] = df_capacity_r["puismaxinstallee"].cumsum()
    df_capacity_r.loc[:, "chronique_capacity"] += baseline_capacity

    # Resample (optionnel)
    df_total_capacity = df_capacity_r["chronique_capacity"].resample(resample_method).transform('mean')/1E3 # type: ignore #MW

    # On filtre sur la fin
    df_total_capacity = df_total_capacity[df_total_capacity.index < pd.Timestamp(end_date)]
    logging.info(f"DataFrame puissance cumulée sur l'intervalle [{start_date}, {end_date}] créé")

    return df_total_capacity
#%%
def resample_to_hourly_ffill(ts: pd.Series) -> pd.Series :
    """Convertis une série temporelle (ts) journalière vers une agrégation horaire en ffill

    Args:
        ts (pd.Series): Série temporelle journalière

    Returns:
        ts_hourly_ffilled (pd.Series): Série temporelle horaire ffilled
    """

    logging.info(f"Démarrage de la conversion jours -> heures, avec interpolation ffill")

    # Simple vérification
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index)
    
    logging.info(f"Données originales: {len(ts)} jours")
    logging.info(f"Période: {ts.index.min().date()} à {ts.index.max().date()}")

    # Daterange
    hourly_index = pd.date_range(start=ts.index.min().floor('D'),
                                 end=ts.index.max().floor('D') + pd.Timedelta(days=1, hours=-1),
                                 freq='h')
    
    logging.info(f"Après conversion, la longueur du dataset est maintenant de {len(hourly_index)}")

    ts_hourly_ffilled = ts.reindex(hourly_index, method="ffill")

    return ts_hourly_ffilled
# %%
def inference_installed_power_sum(df_installed_capacity: pd.DataFrame, region_code: int) -> np.float64:
    """
    Calcule la capacité de production solaire totale installée (en MWh) pour une région donnée,
    après filtrage des valeurs aberrantes temporelles et spécifiques à la région.

    La fonction filtre d'abord les données selon le code de région via `solar_puissance_filter`,
    puis élimine les valeurs temporelles aberrantes grâce à `delete_temporal_outlier`.
    Elle renvoie ensuite la somme des puissances installées convertie en MWh.

    Args:
        df_installed_capacity (pd.DataFrame): 
            DataFrame contenant les informations sur la puissance installée des installations solaires.
            Doit inclure au minimum une colonne `"puismaxinstallee"` (puissance installée en kW)
            et un identifiant de région.
        region_code (int): 
            Code numérique identifiant la région pour laquelle calculer la capacité totale.

    Returns:
        np.float64: 
            Capacité totale installée dans la région spécifiée, exprimée en mégawattheures (MWh).

    Notes:
        - Le calcul divise la somme des puissances installées (en kW) par 1e3 pour obtenir des MWh.
        - Les fonctions `solar_puissance_filter` et `delete_temporal_outlier` doivent être définies ailleurs dans le projet.
    """
        
    # Filtering temporal outlier
    regional_installed_capacity = solar_puissance_filter(df_installed_puissance=df_installed_capacity,
                                            region_code=region_code)
    df_solar_puissance = delete_temporal_outlier(regional_installed_capacity )

    # Somme totale de la capacité installée en MWh
    sum_capacity = df_solar_puissance["puismaxinstallee"].sum()/1E3

    return sum_capacity