# Data
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_pipeline import processors

# Utils
from typing import Tuple, List, Any

def prepare_horizon_data(X: pd.DataFrame, y: pd.Series, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return shift + alignment for a given horizon"""
    
    # Init
    y_shifted = y.shift(-horizon).dropna()
    X_aligned = X.loc[y_shifted.index]

    return X_aligned, y_shifted

def prepare_lstm_data(
    X: pd.DataFrame,
    y: pd.Series,
    meteo_features: List[str],
    encoder_features: List[str],
    decoder_features: List[str],
    seq_length: int,
    output_len: int,
    batch_size: int,
    stride: int,
    val_ratio: float = 0.25) -> Tuple[DataLoader, DataLoader, Any, pd.DataFrame, pd.Series]:
    
    """
    Prepares train/validation DataLoaders for a Seq2Seq LSTM model.
 
    Splits the dataset, fits a preprocessing pipeline on training data, scales features
    and target, and builds sliding-window sequences for encoder-decoder training.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.
        meteo_features (List[str]): Meteorological feature names.
        encoder_features (List[str]): Features used by the encoder.
        decoder_features (List[str]): Features used by the decoder.
        seq_length (int): Length of input sequence.
        output_len (int): Length of prediction horizon.
        batch_size (int): Batch size for DataLoaders.
        stride (int): Step size for sliding windows.
        val_ratio (float, optional): Fraction of data used for validation. Defaults to 0.25.

    Returns:
        Tuple[DataLoader, DataLoader, Any, pd.DataFrame, pd.Series]:
            train_dl: Training DataLoader.
            val_dl: Validation DataLoader.
            processor: Fitted SolarDataProcessor.
    """
 
    # Init
    split_idx = int(len(X) * (1 - val_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
 
    # Preprocessing
    all_lstm_features = list(set(encoder_features) | set(decoder_features))
    processor = processors.SolarDataProcessor(meteo_features=meteo_features, selected_features=all_lstm_features)  
    processor.fit(X_train, y_train)
    
    # Scaling
    X_train_scaled  = processor.transform(X_train)
    X_val_scaled = processor.transform(X_val)
    y_train_scaled  = processor.transform_y(y_train)
    y_val_scaled = processor.transform_y(y_val)
 
    # Dataloaders
    train_dl = to_dataloader(  
        training_dataset=X_train_scaled,
        encoder_features=encoder_features,
        decoder_features=decoder_features,
        target=y_train_scaled,
        seq_length=seq_length,
        seq_future_length=output_len,
        batch_size=batch_size,
        stride=stride,
    )
    val_dl = to_dataloader(
        training_dataset=X_val_scaled,
        encoder_features=encoder_features,
        decoder_features=decoder_features,
        target=y_val_scaled,
        seq_length=seq_length,
        seq_future_length=output_len,
        batch_size=batch_size,
        stride=stride,
    )
    
    # infer_sgignature
    example_end    = seq_length + output_len
    X_example      = X_train.iloc[:example_end]
    y_past_example = y_train.iloc[:seq_length] 
 
    return train_dl, val_dl, processor, X_example, y_past_example

def to_dataloader(
        training_dataset: pd.DataFrame, 
        encoder_features: List[str],
        decoder_features: List[str],
        target: pd.Series,
        seq_length: int,
        seq_future_length: int, 
        batch_size: int,
        stride: int
        ) -> DataLoader:
    
    """Construit un DataLoader Seq2Seq avec séparation encoder/decoder.
    
    Args:
        training_dataset: DataFrame complet des features
        encoder_features: Features passées (autorégressives + cycliques) 
        decoder_features: Features futures connues (météo + cycliques)
        target: Série cible (solar_mw normalisé)
        seq_length: Longueur séquence encodeur
        seq_future_length: Longueur séquence décodeur (horizons)
        batch_size: Taille des batchs
        stride: Pas entre les fenêtres
    
    Returns:
        DataLoader PyTorch

    """

    #Init
    L_total = training_dataset.shape[0]
    total_window_length = seq_length + seq_future_length
    indices = np.arange(0, L_total-total_window_length+1, stride, dtype=int)

    # Découpage des indices
    indices_past = indices[:, None] + np.arange(seq_length) 
    indices_future = indices[:, None] + np.arange(seq_length, (seq_length+seq_future_length)) 
    
    # encoder: (features autorégressives + cycliques + y)
    partial_encoder = training_dataset[encoder_features]
    encoder_dataset = pd.concat([partial_encoder, target], axis=1)
    X_past = torch.tensor(
        encoder_dataset.values[indices_past], dtype=torch.float32
    )

    # eecoder: futur (prévisions + encodages cycliques)
    X_future = torch.tensor(
        training_dataset[decoder_features].values[indices_future], dtype=torch.float32
    )

    # target
    y_target = torch.tensor(
        target.values[indices_future], dtype=torch.float32
    ).unsqueeze(-1)

    # Tensor final
    final_dataset = TensorDataset(X_past, X_future, y_target)
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=False)
    
    # Asserts
    assert X_past.shape[2]   == len(encoder_features) + 1, \
        f"Encoder shape mismatch: {X_past.shape[2]} - {len(encoder_features) + 1}"
    assert X_future.shape[2] == len(decoder_features), \
        f"Decoder shape mismatch: {X_future.shape[2]} - {len(decoder_features)}"
    
    return dataloader