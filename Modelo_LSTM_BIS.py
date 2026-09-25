"""
Modelo_LSTM_BIS.py
==================

Modelo LSTM para prediccion del indice BIS utilizando
las senales de infusion de propofol y remifentanilo.

Arquitectura:
    Dos LSTMs independientes (una por farmaco) cuyos estados ocultos
    finales se concatenan y pasan por un regresor MLP para predecir el BIS.


Entradas:
    x : Tensor (B, T, 2)
        Canal 0 -> Propofol
        Canal 1 -> Remifentanilo

Salida:
    BIS predicho : Tensor (B,)
"""

import torch
import torch.nn as nn


class ModeloLSTMBIS(nn.Module):
    """
    Predictor de BIS con dos LSTMs independientes.

    Parametros
    ----------
    unidades_lstm : int
        Unidades ocultas de cada LSTM (default: 64).
    num_capas : int
        Numero de capas apiladas en cada LSTM (default: 2).
    dropout : float
        Tasa de dropout aplicada entre capas LSTM y en el regresor (default: 0.25).
    """

    def __init__(
        self,
        unidades_lstm: int = 64,
        num_capas: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        lstm_kwargs = dict(
            hidden_size=unidades_lstm,
            num_layers=num_capas,
            batch_first=True,
            dropout=dropout if num_capas > 1 else 0.0,
        )

        # =====================================================
        # LSTM independiente para propofol
        # =====================================================
        self.lstm_propofol = nn.LSTM(input_size=1, **lstm_kwargs)

        # =====================================================
        # LSTM independiente para remifentanilo
        # =====================================================
        self.lstm_remifentanilo = nn.LSTM(input_size=1, **lstm_kwargs)

        # =====================================================
        # Regresor MLP: (unidades*2) -> unidades -> 1
        # =====================================================
        self.regresor = nn.Sequential(
            nn.Linear(unidades_lstm * 2, unidades_lstm),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(unidades_lstm, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametros
        ----------
        x : Tensor (B, T, 2)
            B : tamano de lote
            T : longitud temporal
            Canal 0 : propofol
            Canal 1 : remifentanilo

        Retorna
        -------
        Tensor (B,)  BIS predicho
        """
        propofol = x[:, :, 0:1]   # (B, T, 1)
        remifent = x[:, :, 1:2]   # (B, T, 1)

        _, (h_p, _) = self.lstm_propofol(propofol)
        _, (h_r, _) = self.lstm_remifentanilo(remifent)

        # Ultimo estado oculto de la ultima capa de cada LSTM
        h = torch.cat([h_p[-1], h_r[-1]], dim=1)   # (B, unidades*2)

        return self.regresor(h).squeeze(-1)