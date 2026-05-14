#%%
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Field
from pandera.typing import Series
#%%
class SolarProductionSchema(pa.DataFrameModel):
    """
    Solar production feature derived from RTE API with associated timestamp.
    """

    class Config: # type: ignore
        coerce = True
        strict = True
        name = "SolarProductionSchema"

    date_heure: Series[str] = pa.Field(
        unique=True,
        nullable=False,
        description="timestamp + offset France"
    )

    solaire: Series[float] = pa.Field(
        ge=0.0,
        le=10_000,
        nullable=True,
        description="Instant solar regional production in MWh (0 if night)"
    )
   
# %%
class WeatherPastModel(pa.DataFrameModel):
    """
    NWP weather features for the solar prediction model.
    Each row represents one hour of the forecast.
    The _run_13 columns show the central value;
    _delta_minmax and _std capture the spatial uncertainty of the barycenters
    """
 
    # ------------------------------------------------------------------
    # Horodatage
    # ------------------------------------------------------------------
    date_run_13: Series[pd.Timestamp] = Field(
        unique=True,
        nullable=False,
        description="Timestamp NWP. Hour resolution",
    )
 
    # ------------------------------------------------------------------
    # Temperature (°C)
    # ------------------------------------------------------------------
    temperature_2m_run_13: Series[float] = Field(
        ge=-30.0, le=55.0,
        nullable=False,
        description="Temperature at 2 m — barycentric reference.",
    )
    temperature_2m_delta_minmax: Series[float] = Field(
        ge=0.0, le=30.0,
        nullable=False,
        description="Spatial regional dispersion : min-max temperature (°C).",
    )
    temperature_2m_std: Series[float] = Field(
        ge=0.0, le=20.0,
        nullable=False,
        description="Spatial regional standard deviation of temperature (°C).",
    )
 
    # ------------------------------------------------------------------
    # Relative humidity (%)
    # ------------------------------------------------------------------
    relative_humidity_2m_run_13: Series[float] = Field(
        ge=0.0, le=100.0,
        nullable=False,
        description="Relative humidity at 2 m — reference run (%).",
    )
    relative_humidity_2m_delta_minmax: Series[float] = Field(
        ge=0.0, le=100.0,
        nullable=False,
        description="Spatial regional min-max : relative humidity (%).",
    )
    relative_humidity_2m_std: Series[float] = Field(
        ge=0.0, le=75.0,
        nullable=False,
        description="Spatial regional standard deviation : relative humidity (%).",
    )
 
    # ------------------------------------------------------------------
    # Precipitations (mm/h) — sparse
    # ------------------------------------------------------------------
    precipitation_run_13: Series[float] = Field(
        ge=0.0, le=250.0,
        nullable=True,
        description="Precipitations — reference run (mm/h).",
    )
    precipitation_delta_minmax: Series[float] = Field(
        ge=0.0, le=150.0,
        nullable=True,
        description="Spatial regional min-max : precipitations (mm/h).",
    )
    precipitation_std: Series[float] = Field(
        ge=0.0, le=50.0,
        nullable=True,
        description="Spatial regional standard deviation : precipitations (mm/h).",
    )
 
    # ------------------------------------------------------------------
    # Surface pressure (hPa)
    # ------------------------------------------------------------------
    surface_pressure_run_13: Series[float] = Field(
        ge=800.0, le=1200.0,
        nullable=False,
        description="Surface atmospheric pressure — Reference run (hPa).",
    )
    surface_pressure_delta_minmax: Series[float] = Field(
        ge=0.0, le=200.0,
        nullable=False,
        description=("Regional spatial min-max - surface atmospheric pressure (hPa)"),
    )
    surface_pressure_std: Series[float] = Field(
        ge=0.0, le=50.0,
        nullable=False,
        description=(
            "Spatial regional standard deviation : atmospheric pressure (hPa)")
    )
 
    # ------------------------------------------------------------------
    # Couverture nuageuse (%)
    # ------------------------------------------------------------------
    cloud_cover_run_13: Series[float] = Field(
        ge=0.0, le=100.0,
        nullable=False,
        description="Couverture nuageuse totale — run de référence (%).",
    )
    cloud_cover_delta_minmax: Series[float] = Field(
        ge=0.0, le=100.0,
        nullable=False,
        description="Dispersion inter-runs : max – min de couverture nuageuse (%).",
    )
    cloud_cover_std: Series[float] = Field(
        ge=0.0, le=50.0,
        nullable=False,
        description="Écart-type inter-runs de couverture nuageuse (%).",
    )
 
    # ------------------------------------------------------------------
    # Vitesse du vent à 10 m (km/h)
    # ------------------------------------------------------------------
    wind_speed_10m_run_13: Series[float] = Field(
        ge=0.0, le=300.0,
        nullable=False,
        description="Vitesse du vent à 10 m — run de référence (km/h).",
    )
    wind_speed_10m_delta_minmax: Series[float] = Field(
        ge=0.0, le=200.0,
        nullable=False,
        description="Dispersion inter-runs : max – min de vitesse vent (km/h).",
    )
    wind_speed_10m_std: Series[float] = Field(
        ge=0.0, le=50.0,
        nullable=False,
        description="Écart-type inter-runs de vitesse du vent (km/h).",
    )
 
    # ------------------------------------------------------------------
    # Direction du vent à 10 m (°, convention météo 0–360)
    # ------------------------------------------------------------------
    wind_direction_10m_run_13: Series[float] = Field(
        ge=0.0, le=360.0,
        nullable=False,
        description="Direction du vent à 10 m — run de référence (°).",
    )
    wind_direction_10m_delta_minmax: Series[float] = Field(
        ge=0.0, le=360.0,
        nullable=False,
        description="Dispersion inter-runs : max – min direction vent (°). Peut dépasser 180° si changement de quadrant.",
    )
    wind_direction_10m_std: Series[float] = Field(
        ge=0.0, le=180.0,
        nullable=False,
        description="Écart-type inter-runs de direction du vent (°).",
    )
 
    # ------------------------------------------------------------------
    # Irradiance globale inclinée — GTI (W/m²)
    # ------------------------------------------------------------------
    global_tilted_irradiance_run_13: Series[float] = Field(
        ge=0.0, le=1500.0,   # constante solaire ≈ 1361 W/m² ; marge pour réflexion
        nullable=True,        # nulle la nuit, NaN possible selon fournisseur
        description="Irradiance globale inclinée — run de référence (W/m²).",
    )
    global_tilted_irradiance_delta_minmax: Series[float] = Field(
        ge=0.0, le=1500.0,
        nullable=True,
        description="Dispersion inter-runs : max – min de GTI (W/m²).",
    )
    global_tilted_irradiance_std: Series[float] = Field(
        ge=0.0, le=500.0,
        nullable=True,
        description="Écart-type inter-runs de GTI (W/m²).",
    )

    class Config: # type: ignore
        coerce = True
        strict = True
        name = "WeatherPastModel"

# 