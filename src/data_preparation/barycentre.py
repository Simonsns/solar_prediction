# Ce fichier propose la computation d'un barycentre pondéré par la puissance solaire installée,
# Ce barycentre servira ensuite de scénario central météorologique de la région étudiée,
# Ensuite, d'autres barycentres seront calculés pour chaque département, pour quantifier la dispersion autour du scénario central :
# variance, moyenne dans la région étudiée. Elles seront prises comme features dans les modèles non convolutifs.
#%%
import pandas as pd
import geopandas as gpd
import numpy as np
#%%
def compute_barycentre(lat: np.ndarray, long: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calcule le barycentre pondéré par un poids pour une zone donnée

    Args:
        lat (np.ndarray): Toutes les latitudes des objets qui possèdent un poids
        long (np.ndarray):  Toutes les longitudes des objets qui possèdent un poids
        weights (np.ndarray): Poids assignés aux objets

    Returns:
        np.ndarray: latitude et longitude du barycentre
    """

    return np.array([np.sum(lat*weights)/np.sum(weights), np.sum(long*weights)/np.sum(weights)])