# Data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# utils
from typing import List, Optional


class  SolarDataProcessor(BaseEstimator, TransformerMixin):
    
    """Data preprocessing before model inference :
    - Target features normalization by the 99th quantile to detrend the time serie (last quantile for validation/prod and series for training);
    - Scaling (MinMax) the others features to increase the convergence (if used)
    """

    def __init__(
            self,  
            meteo_features: List[str], 
            window_days: float = 90, 
            quantile: float = 0.99,
            annot: str = "mw",
            selected_features: Optional[List[str]] = None
            ):
        
        self.meteo_features = meteo_features
        self.window_days = window_days
        self.quantile = quantile
        self.meteo_scaler = MinMaxScaler()
        self.annot = annot
        self.selected_features = selected_features

    def _rolling_quantile(
            self, 
            y: pd.Series) -> pd.Series:
        
        """Return the 99th rolling quantile on window days"""
        
        window_hours = int(self.window_days * 24)

        return y.rolling(window=window_hours, min_periods=24).quantile(self.quantile).bfill().astype(np.float32)
        

    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        # Meteo features fit
        self.effective_meteo_features_ = [f for f in self.meteo_features if f in X.columns]
        if self.effective_meteo_features_:
            self.meteo_scaler.fit(X[self.effective_meteo_features_])

        if y is not None:   
            # Quantile fit
            full_denominators = self._rolling_quantile(y)
            self.denominator_series_ = full_denominators[~full_denominators.index.duplicated(keep='last')] # Eviter les doubles index
            self.last_denominator_ = float(full_denominators.iloc[-1])

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_res = X.copy()

        # MinMax meteo features scaling
        to_scale = [f for f in self.effective_meteo_features_ if f in X.columns]
        if to_scale:
            X_res[to_scale] = self.meteo_scaler.transform(X[to_scale])

        # rolling quantile if not training, last quantile if not
        denoms = (
            self.denominator_series_
            .reindex(X_res.index)
            .fillna(self.last_denominator_)
            .values
        )
            
        # Derivated target features scaling
        mw_columns = [col for col in X_res.columns if self.annot in col]
        for col in mw_columns:
            X_res[col] = X_res[col] / denoms
        
        
        if self.selected_features is not None:
            return X_res[[c for c in self.selected_features if c in X_res.columns]]   
        
        return X_res
    
    def transform_y(self, y: pd.Series) -> pd.Series:
        
        denoms = (
            self.denominator_series_
            .reindex(y.index)
            .fillna(self.last_denominator_)
            .values)
        
        return y / denoms
        
    def inverse_transform_y(self, y_pred: np.ndarray, input_index: pd.Index) -> np.ndarray:
        """Return y in MWh. Must the index of infered or trained input to reindex quantiles"""
        
        denoms = (
            self.denominator_series_
            .reindex(input_index)
            .fillna(self.last_denominator_)
            .values
        )
        return y_pred * denoms