import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel
from typing import Dict, List
from sklearn.pipeline import Pipeline
import torch.nn as nn
import torch

class MultiHorizonLGBMWrapper(PythonModel):
    
    def __init__(self, models_dict: Dict[str, Pipeline], num_horizons: int):
        self.models_dict = models_dict
        self.num_horizons = num_horizons
        
    def predict(self, model_input: pd.DataFrame): # type: ignore

        all_predictions = {}
        future_index = model_input.index

        for h in range(1, self.num_horizons+1):
            key = f"h+{h}"
            if key in self.models_dict:
                pipe = self.models_dict[key]
                norm_pred = pipe.predict(model_input)
                all_predictions[key] = pipe.named_steps['processor'].inverse_transform_y(norm_pred, future_index)
    
        return pd.DataFrame(all_predictions, index=model_input.index).clip(lower=0)


class SolarLSTMWrapper(PythonModel):
    """Wrapper PythonModel pour LSTM Seq2Seq"""
 
    def __init__(
            self, 
            processor, 
            model: nn.Module, 
            encoder_features: List[str],
            decoder_features: List[str],
            seq_length: int,
            output_len: int,
            device: str = "cpu"       
    ) -> None:
        
        self.processor        = processor
        self.model            = model
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.seq_length       = seq_length
        self.output_len       = output_len
        self.device           = torch.device(device)
    
    def _run_inference(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """Isolated forward pass"""
        self.model.eval()
        return self.model(enc, dec)
    
    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        
        """Prédit 24 horizons à partir d'une séquence encoder+decoder.
 
        Args:
        model_input: DataFrame TOTAL (y inclus)
 
        Returns:
        - preds (np.ndarray): predictions LSTM

        """
        if len(model_input) != (self.seq_length + self.output_len):
            raise ValueError(f"Input must have {self.seq_length + self.output_len} rows, got {len(model_input)}")
        
        # Init
        y_past = model_input['solar_mw'].iloc[:self.seq_length]
        X = model_input.drop(columns=["solar_mw"])
        
        # Scaling 
        X_scaled = self.processor.transform(X)

        # Slicing
        X_past_scaled   = X_scaled[self.encoder_features].iloc[:self.seq_length]
        X_future_scaled = X_scaled[self.decoder_features].iloc[self.seq_length:]

        # y_past ajouté à l'encodeur (normalisé)
        y_past_series = pd.Series(y_past.values, index=X.index[:self.seq_length], name="solar_mw")
        y_past_scaled = self.processor.transform_y(y_past_series)
        
        enc_array = np.concatenate(
            [X_past_scaled.values, y_past_scaled.values.reshape(-1, 1)], axis=1
        )

        # Inférence
        enc = torch.tensor(enc_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        dec = torch.tensor(X_future_scaled.values, dtype=torch.float32).unsqueeze(0).to(self.device)
        norm_preds = self._run_inference(enc, dec)   # (1, output_len, 1)
        norm_preds_np = norm_preds.squeeze().cpu().numpy()

        # Denormalization
        preds = self.processor.inverse_transform_y(norm_preds_np, X_future_scaled.index)

        # Return
        return np.clip(preds, 0, None)