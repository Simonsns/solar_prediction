# Data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import io
from supabase import create_client, Client

# utils
from typing import List, Optional, Dict, Any, Tuple

class SolarDataProcessor(BaseEstimator, TransformerMixin):
    
    """Data preprocessing before model inference :
    - Target features normalization by the 99th quantile to detrend the time serie (last quantile for validation/prod and series for training);
    - Scaling (MinMax) the others features to increase the convergence (if used)
    """

    def __init__(
            self,  
            meteo_features: List[str], 
            window_days: float = 90, 
            quantile: float = 0.99,
            annot: str = "solaire", 
            ):
        
        self.meteo_features = meteo_features
        self.window_days = window_days
        self.quantile = quantile
        self.meteo_scaler = MinMaxScaler()
        self.annot = annot

    def _rolling_quantile(
            self, 
            y: pd.Series) -> pd.Series:
        
        """Return the 99th rolling quantile on window days"""
        
        window_hours = int(self.window_days * 24)

        return ( y
                .rolling(window=window_hours, min_periods=24)
                .quantile(self.quantile)
                .bfill()
                .astype(np.float32)
        )
        
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
        annot_columns = [col for col in X_res.columns if self.annot in col]
        for col in annot_columns:
            X_res[col] = X_res[col] / denoms
        
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

class LGBMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Agnostic feature selector based on gain importance of LGBM.
    Select variables according to Pareto front on several temporal horizon.
    """

    def __init__(
            self,
            lgbm_params: Dict[str, Any],
            horizons: List[int],
            threshold: float = 0.95,
            always_keep: Optional[List[str]] = None
        ):
        # initialization
        self.lgbm_params = lgbm_params
        self.horizons = horizons
        self.threshold = threshold
        self.always_keep = always_keep or []

        # Dynamic learned parameters
        self.selected_features_per_horizon_: Dict[str, List[str]] = {}
        self.final_selected_features_: List[str] = []
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train a LGBM model for each horizon to memorize best features set respectively"""

        all_selected_set = set()

        for horizon in self.horizons:
            
            # Temporal alignment
            y_shifted = y.shift(-horizon).dropna()
            X_aligned = X.loc[y_shifted.index]

            # Training
            model = lgb.LGBMRegressor(**self.lgbm_params)
            model.fit(X=X_aligned, y=y_shifted)

            # Gain
            gain_df = ( 
                pd.DataFrame({'Feature': X.columns, 'Gain': model.feature_importances_})
                .sort_values(by="Gain", ascending=False)
            )
            gain_df["cumulative_gain"] = (gain_df["Gain"] / gain_df["Gain"].sum()).cumsum()
            
            # Pareto selector
            selected_cols = (
                gain_df.loc[gain_df["cumulative_gain"] < self.threshold, "Feature"]
                .to_list()
            )

            # Metadata stockage
            self.selected_features_per_horizon_[f"h_{horizon}"] = selected_cols
            all_selected_set.update(selected_cols)

        # Needed features
        features_kept = [f for f in self.always_keep if f in X.columns]
        all_selected_set.update(features_kept)

        # Stateful transformer
        self.final_selected_features_ = list(all_selected_set)
        self._is_fitted = True

        return self     

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select useful variables on a new DataFrame"""
        if not self._is_fitted:
            raise ValueError("Selector has not bee fitted. Call .fit() before transform values")
        
        return X[self.final_selected_features_]

class LGBMDataloader:

    def __init__(self, url: str, key: str, test_size: float = 0.2):
        self.url = url
        self.key = key
        self.test_size = test_size
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Lazy initialization of Supabase Client"""
        if self._client is None:
            self._client = create_client(self.url, self.key)
        
        return self._client
    
    def download_parquet(
        self,
        bucket_name: str,
        file_path: str,
        ) -> pd.DataFrame:
        """
        Download a parquet file from a bucket in Supabase and return a DataFrame.
        """

        try:
            logging.info(f"Downloading artifact from Supabase...")
        
            response_bytes = self.client.storage.from_(bucket_name).download(path=file_path)
            df = pd.read_parquet(io.BytesIO(response_bytes))
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Paris")
            logging.info(f"[SUCCESS] Downloaded artifact : {len(df)} rows, {len(df.columns)} columns.")
            
            return df

        except Exception as e:
            logging.error(f"[ERROR] Fail downloading artifact from Supabase : {str(e)}")
            raise RuntimeError(f"Cannot proceed to artifact recuperation  : {e}")
    
    def split_dataframe(
            self,
        training_dataset: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        if "solaire" not in training_dataset.columns:
            raise ValueError("Target feature *solaire* is missing")
    
        # Features
        X = training_dataset.drop(columns="solaire")
        y = training_dataset["solaire"]

        # Séparation train test split
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.test_size,
                                                            shuffle=False, 
                                                            random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def run(self, 
            bucket_name: str,
            file_path: str,
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and split training DataFrame orchestration"""
        
        df = self.download_parquet(
            bucket_name=bucket_name,
            file_path=file_path)
        X_train, X_test, y_train, y_test = self.split_dataframe(training_dataset=df)
        
        return  X_train, X_test, y_train, y_test
        