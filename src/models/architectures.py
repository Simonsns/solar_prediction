### LSTM Seq2Seq Architecture

import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        #LayerNorm
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x_past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #x de la forme (batch_number, seq_length, all_features)
        _, (h_n, c_n) = self.encoder(x_past)
        h_n_norm = self.layer_norm(h_n) 
        c_n_norm = self.layer_norm(c_n)

        return (h_n_norm, c_n_norm)

class Decoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_len: int, dropout: float):
        
        super(Decoder, self).__init__()
        self.output_len = output_len # M=24

        #Decoder
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        #Couche linéaire pour la sortie en prediction
        self.fc_out = nn.Linear(hidden_size, 1)


    def forward(self, x_future: torch.Tensor, h_init_norm: torch.Tensor, c_init_norm: torch.Tensor) -> torch.Tensor:
        #x_future de la forme (batch_number, output_len, feature.difference(y))
        output, _ = self.decoder(x_future, (h_init_norm, c_init_norm))

        #Couche linéaire
        predictions = self.fc_out(output)

        return predictions
    
class Seq2seq(nn.Module):
    
    def __init__(
            self, 
            past_features_length: int, 
            future_features_length: int, 
            hidden_size: int, 
            num_layers: int, 
            output_len: int, 
            dropout: float = 0.0
        ):
        
        super(Seq2seq, self).__init__()
        
        self.encoder = Encoder(
            input_size=past_features_length, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            input_size=future_features_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_len=output_len,
            dropout=dropout
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        #Encoder
        (h_n_norm, c_n_norm) = self.encoder(x_past)
        predictions = self.decoder(x_future, h_n_norm, c_n_norm)

        return predictions