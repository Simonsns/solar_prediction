# Data contract
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Any
import mlflow
import pandas as pd
import torch.nn as nn
import numpy as np

# ---LGBM contract---
@dataclass
class HorizonRunResult:
    """Data contract between ML and MLflow logs"""
    # Identifiers
    horizon:  int
    set_name: str
    
    # ML results
    model:        Pipeline
    best_rmse:    float
    best_params:  dict
    X_train:      pd.DataFrame

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]
    
    @property
    def params_to_log(self) -> Dict[str, Optional[float|str]]:
        """HP and experimental context (MLflow log_params)"""

        return {
            **self.best_params,
            "horizon":     self.horizon,
            "feature_set": self.set_name,
            "n_features" : self.n_features,
        }

    @property
    def tags_to_log(self) -> Dict[str, str]:
        """
        Tags to log to MLFlow server
        """

        return {
            "project"       : "solar_forecast",
            "model_family"  : "lightgbm",
            "training_type" : "multihorizon",
            "horizon_bucket":  self._horizon_bucket(),
            "feature_set"   :  self.set_name,
            "is_baseline"   :  str(self.set_name == "short"),
        }
    
    @property
    def signature(self):
        return mlflow.models.infer_signature( # type: ignore
            self.X_train,
            self.model.predict(self.X_train)
        )
    
    def to_record(self) -> Dict[str, Optional[float|str]]:
        """Return results dict"""
        return {
            "horizon":     self.horizon,
            "feature_set": self.set_name,
            "n_features":  self.n_features,
            "rmse":        self.best_rmse,
            **self.best_params,
        }
    
    def _horizon_bucket(self) -> str:
        """Easy filter"""
        if self.horizon <= 6:   
            return "short"
        if self.horizon <= 12:  
            return "medium"
        return "long"

# ---LSTM contract---
@dataclass
class LSTMRunResult:
    """Data contract between LSTM and MLflow"""
 
    # Params
    num_epochs:  int
    seq_length:  int    # longueur encodeur
    output_len:  int    # longueur décodeur = nb horizons
    model:        nn.Module
    best_rmse:    float
    best_params:  Dict[str, Any]
    encoder_features: List[str]
    decoder_features: List[str]
    processor: Any
 
    @property
    def params_to_log(self) -> Dict[str, Any]:
        return {
            **self.best_params,
            "num_epochs": self.num_epochs,
            "seq_length": self.seq_length,
            "output_len": self.output_len,
        }
 
    @property
    def tags_to_log(self) -> Dict[str, str]:
        return {
            "project":       "solar_forecast",
            "model_family":  "LSTMSeq2Seq",
            "framework":     "pytorch",
            "training_type": "seq2seq",
        }
 
    @property
    def signature(self) -> mlflow.models.ModelSignature: # type: ignore
        
        # Def features
        all_features = list(set(self.encoder_features + self.decoder_features))
        input_cols = [mlflow.types.ColSpec("double", col) for col in all_features] # type: ignore
        input_cols.append(mlflow.types.ColSpec("double", "solar_mw")) # type: ignore
        
        # Inputs cols
        inputs = mlflow.types.Schema(input_cols) # type: ignore

        outputs = mlflow.types.Schema([ # type: ignore
            mlflow.types.TensorSpec( # type: ignore
                type=np.dtype("float32"),  
                shape=(-1, self.output_len), 
                name="prediction_mw"
            )
        ])

        return mlflow.models.ModelSignature(inputs=inputs, outputs=outputs) # type: ignore