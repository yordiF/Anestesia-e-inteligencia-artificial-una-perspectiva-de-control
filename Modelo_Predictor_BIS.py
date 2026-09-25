
"""
Modelo_Predictor_BIS.py
=======================
Archivo importable con la arquitectura completa del modelo de predicción
del índice biespectral (BIS) durante anestesia TIVA.


Arquitectura:
    Bloque A : CNNCinetica → GRNCinetica → CNNDinamica → EstimadorBISPseudo
    Bloque B : LSTM × 3 (propofol, remifentanilo, BIS pseudo) + GRN fusión
    Bloque C : AtencionMulticabeza interpretable + Bottleneck

Clases exportadas:
    ModeloAnestesia      ← clase principal; única que necesitas importar
    EstimadorBISPseudo
    CNNCinetica
    GRNCinetica
    CNNDinamica
    GRN
    AtencionMulticabeza
"""

import math
import torch
import torch.nn as nn

# =============================================================================
# VALORES POR DEFECTO DE LOS HIPERPARÁMETROS
# =============================================================================
_UNIDADES_LSTM    = 64
_UNIDADES_GRN     = 64
_CABEZAS_ATENCION = 4
_DROPOUT          = 0.35

# CNN cinética
_FILTROS_CIN_C1   = 16
_KERNEL_CIN_C1    = 5

# CNN dinámica
_FILTROS_DIN_C1   = 12
_KERNEL_DIN_C1    = 9
_FILTROS_DIN_C2   = 6
_KERNEL_DIN_C2    = 5


# =============================================================================
# GRN  —  Gated Residual Network
# =============================================================================
class GRN(nn.Module):
    """
    Red Residual con Compuerta  del Temporal Fusion
    Transformer .

    Fusiona una entrada primaria 'a' con un contexto externo opcional 'c'
    mediante un mecanismo de compuerta (GLU) y una conexión residual.

    Parámetros
    ----------
    dim_entrada  : dimensión de la entrada primaria a
    dim_oculta   : dimensión de las capas internas
    dim_salida   : dimensión de la salida
    dim_contexto : dimensión del vector de contexto c (None → sin contexto)
    tasa_dropout : tasa de abandono aplicada antes del GLU
    """

    def __init__(self, dim_entrada, dim_oculta, dim_salida,
                 dim_contexto=None, tasa_dropout=_DROPOUT):
        super().__init__()
        self.fc1       = nn.Linear(dim_entrada, dim_oculta)
        self.fc2       = nn.Linear(dim_oculta,  dim_oculta)
        self.fc_salida = nn.Linear(dim_oculta,  dim_salida * 2)   # entrada al GLU
        self.norm      = nn.LayerNorm(dim_salida)
        self.dropout   = nn.Dropout(tasa_dropout)
        self.elu       = nn.ELU()

        # Proyección lineal del contexto externo (covariables del paciente)
        self.fc_contexto = (
            nn.Linear(dim_contexto, dim_oculta, bias=False)
            if dim_contexto is not None else None
        )
        # Proyección residual cuando las dimensiones de entrada y salida difieren
        self.proyeccion_residual = (
            nn.Linear(dim_entrada, dim_salida, bias=False)
            if dim_entrada != dim_salida else None
        )

    def forward(self, a, c=None):
        """
        a : (B, T, d) o (B, d)   — entrada primaria
        c : (B, d_ctx)           — covariables estáticas del paciente 
        """
        residual = a
        h        = self.fc1(a)

        if c is not None and self.fc_contexto is not None:
            c_proy = self.fc_contexto(c)
            if a.dim() == 3 and c.dim() == 2:
                c_proy = c_proy.unsqueeze(1)    # (B, 1, d) → broadcast en T
            h = h + c_proy

        h = self.dropout(self.fc2(self.elu(h)))

        # Unidad Lineal con Compuerta (GLU)
        sal_raw   = self.fc_salida(h)
        valor     = sal_raw[..., : sal_raw.shape[-1] // 2]
        compuerta = torch.sigmoid(sal_raw[..., sal_raw.shape[-1] // 2 :])
        h_comp    = valor * compuerta

        if self.proyeccion_residual is not None:
            residual = self.proyeccion_residual(residual)

        return self.norm(h_comp + residual)


# =============================================================================
# BLOQUE A — CNN CINÉTICA
# =============================================================================
class CNNCinetica(nn.Module):
    """
    Estima la concentración en el sitio efector (Ce bruta) a partir de
    los historiales de infusión de propofol y remifentanilo.

    Arquitectura:
        Conv1d(2 → F_C1, k=K_C1, padding='same') + BatchNorm1d + ELU
        Conv1d(F_C1 → 1, k=1)   ← proyección puntual a Ce

    Entrada : (B, 2, T)
    Salida  : (B, T)   — Ce bruta (antes del refinamiento GRN)
    """

    def __init__(self,
                 filtros_c1=_FILTROS_CIN_C1,
                 kernel_c1 =_KERNEL_CIN_C1):
        super().__init__()
        padding    = kernel_c1 // 2
        self.capa1 = nn.Sequential(
            nn.Conv1d(2, filtros_c1, kernel_size=kernel_c1,
                      padding=padding, bias=False),
            nn.BatchNorm1d(filtros_c1),
            nn.ELU(),
        )
        self.capa2 = nn.Conv1d(filtros_c1, 1, kernel_size=1)

    def forward(self, x):
        """x : (B, 2, T)  →  Ce_bruta : (B, T)"""
        return self.capa2(self.capa1(x)).squeeze(1)


# =============================================================================
# BLOQUE A — GRN CINÉTICA  (refinamiento de Ce con covariables)
# =============================================================================
class GRNCinetica(nn.Module):
    """
    Refina la serie Ce producida por CNNCinetica incorporando las covariables
    fisiológicas del paciente como contexto externo (Input 2 del GRN).

    La Ce bruta actúa como entrada primaria (Input 1 / conexión residual)
    y el vector de covariables actúa como contexto externo (Input 2).

    Entrada:
        ce_bruta    : (B, T)
        covariables : (B, 4)
    Salida:
        ce_refinada : (B, T)
    """

    def __init__(self,
                 dim_oculta  =_UNIDADES_GRN,
                 dim_contexto=4,
                 tasa_dropout=_DROPOUT):
        super().__init__()
        # GRN escalar por paso temporal (dim_entrada=1, dim_salida=1)
        self.grn = GRN(
            dim_entrada  = 1,
            dim_oculta   = dim_oculta,
            dim_salida   = 1,
            dim_contexto = dim_contexto,
            tasa_dropout = tasa_dropout,
        )

    def forward(self, ce_bruta, covariables):
        """
        ce_bruta    : (B, T)
        covariables : (B, 4)
        Retorna ce_refinada : (B, T)
        """
        ce_seq     = ce_bruta.unsqueeze(-1)           # (B, T, 1)
        ce_ref_seq = self.grn(ce_seq, covariables)    # (B, T, 1)
        return ce_ref_seq.squeeze(-1)                 # (B, T)


# =============================================================================
# BLOQUE A — CNN DINÁMICA
# =============================================================================
class CNNDinamica(nn.Module):
    """
    Estima los parámetros escalares [W0, W1, W2, W3] de la ecuación del BIS
    pseudo-histórico a partir del historial de valores BIS del paciente.

    Arquitectura:
        Conv1d(1 → F_C1, k=K_C1) + BatchNorm1d + ReLU
        Conv1d(F_C1 → F_C2, k=K_C2) + BatchNorm1d + ReLU
        AdaptiveAvgPool1d(1) → Flatten
        MLP : F_C2 → F_C2 → 4

    Entrada : (B, T)
    Salida  : (B, 4)   — parámetros [W0, W1, W2, W3]
    """

    def __init__(self,
                 filtros_c1=_FILTROS_DIN_C1,
                 kernel_c1 =_KERNEL_DIN_C1,
                 filtros_c2=_FILTROS_DIN_C2,
                 kernel_c2 =_KERNEL_DIN_C2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, filtros_c1, kernel_size=kernel_c1,
                      padding=kernel_c1 // 2, bias=False),
            nn.BatchNorm1d(filtros_c1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(filtros_c1, filtros_c2, kernel_size=kernel_c2,
                      padding=kernel_c2 // 2, bias=False),
            nn.BatchNorm1d(filtros_c2),
            nn.ReLU(),
        )
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.aplanar = nn.Flatten()
        self.mlp     = nn.Sequential(
            nn.Linear(filtros_c2, filtros_c2),
            nn.ReLU(),
            nn.Linear(filtros_c2, 4),
        )

    def forward(self, bis_hist):
        """bis_hist : (B, T)  →  params : (B, 4)"""
        x = bis_hist.unsqueeze(1)                 # (B, 1, T)
        h = self.pool(self.conv2(self.conv1(x)))  # (B, F_C2, 1)
        return self.mlp(self.aplanar(h))          # (B, 4)


# =============================================================================
# BLOQUE A — ESTIMADOR BIS PSEUDO-HISTÓRICO
# =============================================================================
class EstimadorBISPseudo(nn.Module):
    """
    Produce la serie BIS pseudo-histórico que sustituye al modelo PK-PD.

    Flujo completo del Bloque A:
        1. CNNCinetica : infusiones (B,2,T) → Ce_bruta (B,T)
        2. GRNCinetica : Ce_bruta + covariables → Ce_refinada (B,T)
        3. CNNDinamica : bis_hist (B,T) → [W0,W1,W2,W3] (B,4)
        4. BIS_k = W0 + W1 · sigmoid(W2 · Ce_refinada_k + W3)

    Entradas:
        infusiones  : (B, 2, T)
        bis_hist    : (B, T)
        covariables : (B, 4)
    Retorna:
        bis_pseudo  : (B, T)   — BIS estimado en la ventana completa
        ce_refinada : (B, T)   — Ce tras el refinamiento GRN
    """

    def __init__(self,
                 filtros_cin_c1 =_FILTROS_CIN_C1,
                 kernel_cin_c1  =_KERNEL_CIN_C1,
                 filtros_din_c1 =_FILTROS_DIN_C1,
                 kernel_din_c1  =_KERNEL_DIN_C1,
                 filtros_din_c2 =_FILTROS_DIN_C2,
                 kernel_din_c2  =_KERNEL_DIN_C2,
                 dim_covariables=4,
                 dim_oculta_grn =_UNIDADES_GRN,
                 tasa_dropout   =_DROPOUT):
        super().__init__()
        self.cnn_cinetica = CNNCinetica(
            filtros_c1=filtros_cin_c1,
            kernel_c1 =kernel_cin_c1,
        )
        self.grn_cinetica = GRNCinetica(
            dim_oculta   =dim_oculta_grn,
            dim_contexto =dim_covariables,
            tasa_dropout =tasa_dropout,
        )
        self.cnn_dinamica = CNNDinamica(
            filtros_c1=filtros_din_c1,
            kernel_c1 =kernel_din_c1,
            filtros_c2=filtros_din_c2,
            kernel_c2 =kernel_din_c2,
        )

    def forward(self, infusiones, bis_hist, covariables):
        ce_bruta    = self.cnn_cinetica(infusiones)               # (B, T)
        ce_refinada = self.grn_cinetica(ce_bruta, covariables)    # (B, T)

        params         = self.cnn_dinamica(bis_hist)              # (B, 4)
        W0, W1, W2, W3 = (params[:, i:i+1] for i in range(4))

        bis_pseudo = W0 + W1 * torch.sigmoid(W2 * ce_refinada + W3)  # (B, T)
        return bis_pseudo, ce_refinada


# =============================================================================
# BLOQUE C — ATENCIÓN MULTICABEZA INTERPRETABLE
# =============================================================================
class AtencionMulticabeza(nn.Module):
    """
    Atención multicabeza interpretable.

    Los pesos de valor W_V son compartidos entre todas las cabezas;
    los pesos Q y K son propios de cada cabeza.
    Las salidas de las cabezas se promedian en lugar de concatenarse.
    """

    def __init__(self, dim_modelo, num_cabezas, tasa_dropout=_DROPOUT):
        super().__init__()
        assert dim_modelo % num_cabezas == 0
        self.num_cabezas = num_cabezas
        self.dim_cabeza  = dim_modelo // num_cabezas

        self.pesos_Q = nn.ModuleList([
            nn.Linear(dim_modelo, self.dim_cabeza, bias=False)
            for _ in range(num_cabezas)
        ])
        self.pesos_K = nn.ModuleList([
            nn.Linear(dim_modelo, self.dim_cabeza, bias=False)
            for _ in range(num_cabezas)
        ])
        self.pesos_V  = nn.Linear(dim_modelo, dim_modelo, bias=False)
        self.pesos_WH = nn.Linear(dim_modelo, dim_modelo, bias=False)
        self.dropout  = nn.Dropout(tasa_dropout)
        self.escala   = math.sqrt(self.dim_cabeza)

    def forward(self, Q, K, V):
        """Q, K, V : (B, T, dim_modelo)  →  (B, T, dim_modelo)"""
        V_proy = self.pesos_V(V)
        A_suma = None
        for h in range(self.num_cabezas):
            puntajes = torch.bmm(
                self.pesos_Q[h](Q),
                self.pesos_K[h](K).transpose(1, 2)
            ) / self.escala
            A_h    = self.dropout(torch.softmax(puntajes, dim=-1))
            A_suma = A_h if A_suma is None else A_suma + A_h

        A_prom = A_suma / self.num_cabezas
        return self.pesos_WH(torch.bmm(A_prom, V_proy))


# =============================================================================
# MODELO COMPLETO
# =============================================================================
class ModeloAnestesia(nn.Module):
    """
    Modelo completo de predicción del índice BIS durante anestesia TIVA.

    Entradas del método forward:
        infusiones  : (B, 2, T)  float32  — tasas de infusión normalizadas
        bis_hist    : (B, T)     float32  — historial BIS normalizado
        covariables : (B, 4)     float32  — [Edad_norm, Peso_norm,
                                             Altura_norm, Sexo_norm]
    Retorna:
        bis_pred   : (B,)    — predicción BIS en t+1
        bis_pseudo : (B, T)  — serie BIS estimada por el Bloque A
    """

    def __init__(self,
                 unidades_lstm   =_UNIDADES_LSTM,
                 unidades_grn    =_UNIDADES_GRN,
                 num_cabezas     =_CABEZAS_ATENCION,
                 dim_covariables =4,
                 tasa_dropout    =_DROPOUT,
                 filtros_cin_c1  =_FILTROS_CIN_C1,
                 kernel_cin_c1   =_KERNEL_CIN_C1,
                 filtros_din_c1  =_FILTROS_DIN_C1,
                 kernel_din_c1   =_KERNEL_DIN_C1,
                 filtros_din_c2  =_FILTROS_DIN_C2,
                 kernel_din_c2   =_KERNEL_DIN_C2):
        super().__init__()

        # ── Bloque A ──────────────────────────────────────────────────────
        self.estimador_bis_pseudo = EstimadorBISPseudo(
            filtros_cin_c1  =filtros_cin_c1,
            kernel_cin_c1   =kernel_cin_c1,
            filtros_din_c1  =filtros_din_c1,
            kernel_din_c1   =kernel_din_c1,
            filtros_din_c2  =filtros_din_c2,
            kernel_din_c2   =kernel_din_c2,
            dim_covariables =dim_covariables,
            dim_oculta_grn  =unidades_grn,
            tasa_dropout    =tasa_dropout,
        )

        # ── Bloque B — tres módulos LSTM independientes ───────────────────
        self.lstm_propofol   = nn.LSTM(1, unidades_lstm, batch_first=True)
        self.lstm_remifent   = nn.LSTM(1, unidades_lstm, batch_first=True)
        self.lstm_bis_pseudo = nn.LSTM(1, unidades_lstm, batch_first=True)

        self.grn_fusion = GRN(
            dim_entrada  =unidades_lstm * 3,
            dim_oculta   =unidades_grn,
            dim_salida   =unidades_grn,
            tasa_dropout =tasa_dropout,
        )
        self.grn_covariables = GRN(
            dim_entrada  =unidades_grn,
            dim_oculta   =unidades_grn,
            dim_salida   =unidades_grn,
            dim_contexto =dim_covariables,
            tasa_dropout =tasa_dropout,
        )

        # ── Bloque C ──────────────────────────────────────────────────────
        self.atencion      = AtencionMulticabeza(
            unidades_grn, num_cabezas, tasa_dropout
        )
        self.norm_atencion = nn.LayerNorm(unidades_grn)

        self.bottleneck = nn.Sequential(
            nn.Linear(unidades_grn + unidades_lstm * 3, unidades_grn),
            nn.ELU(),
            nn.Dropout(tasa_dropout),
            nn.Linear(unidades_grn, 1),
        )

    def forward(self, infusiones, bis_hist, covariables):
        # ── Bloque A ──────────────────────────────────────────────────────
        bis_pseudo, _ = self.estimador_bis_pseudo(
            infusiones, bis_hist, covariables
        )

        # ── Bloque B ──────────────────────────────────────────────────────
        prop_seq = infusiones[:, 0:1, :].permute(0, 2, 1)   # (B, T, 1)
        remi_seq = infusiones[:, 1:2, :].permute(0, 2, 1)
        bpse_seq = bis_pseudo.unsqueeze(-1)                   # (B, T, 1)

        sal_p, (h_p, _) = self.lstm_propofol(prop_seq)
        sal_r, (h_r, _) = self.lstm_remifent(remi_seq)
        sal_b, (h_b, _) = self.lstm_bis_pseudo(bpse_seq)

        fusion = torch.cat([sal_p, sal_r, sal_b], dim=-1)    # (B, T, 3H)
        z      = self.grn_fusion(fusion)                      # (B, T, Ug)
        z      = self.grn_covariables(z, covariables)         # (B, T, Ug)

        # ── Bloque C ──────────────────────────────────────────────────────
        A      = self.atencion(z, z, z)
        A      = self.norm_atencion(A + z)
        beta_t = A[:, -1, :]                                  # (B, Ug)

        estados = torch.cat(
            [h_p.squeeze(0), h_r.squeeze(0), h_b.squeeze(0)], dim=-1
        )                                                      # (B, 3H)

        bis_pred = self.bottleneck(
            torch.cat([beta_t, estados], dim=-1)
        ).squeeze(-1)                                          # (B,)

        return bis_pred, bis_pseudo