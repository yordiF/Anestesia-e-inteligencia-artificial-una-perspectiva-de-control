"""
Normalizacion_BIS.py
====================
Funciones comunes para la construcción y la normalización de las ventanas
de entrada de los modelos predictores del BIS (RIDA y LSTM de referencia).
Este archivo se importa desde todas las libretas que generan ventanas, ya
sea para el entrenamiento, para la evaluación o para la simulación en lazo
cerrado, de modo que el tratamiento de las entradas sea idéntico en todos
los casos.

Criterios adoptados
-------------------
1. Normalización global. Cada señal (tasa de infusión de propofol, tasa de
   infusión de remifentanilo y BIS) se estandariza con una media y una
   desviación estándar fijas, calculadas una sola vez sobre el conjunto de
   entrenamiento:

        x_norm = (x - media) / desviacion

   Dado que los mismos parámetros se aplican a todas las ventanas, la
   transformación conserva la magnitud de las infusiones y el nivel del
   BIS: una infusión del doble de magnitud produce una entrada distinta, y
   un historial de BIS en torno a 30 se distingue de uno en torno a 60.

2. Relleno. Cuando el historial disponible es menor que la longitud de la
   ventana, las posiciones faltantes de las infusiones se rellenan con cero
   (ausencia de infusión) y las del BIS con el primer valor disponible del
   registro. El relleno se aplica sobre las señales en sus unidades
   originales y antes de la normalización, de manera idéntica en el
   entrenamiento, en la evaluación y en la simulación.

3. Alineación temporal. La ventana asociada al instante n contiene las
   muestras n-L+1, ..., n, y la etiqueta correspondiente es BIS[n+1]. De
   este modo, el modelo predice el valor siguiente a partir de información
   disponible hasta el instante actual, tal como ocurre en el lazo cerrado.
"""

import json
import math
import os

import numpy as np

try:
    import torch
except ImportError:          # El módulo puede emplearse sin PyTorch
    torch = None


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Archivo en el que se almacenan las estadísticas de normalización
RUTA_ESTADISTICAS = r"C:\Users\yordi\Desktop\estadisticas_normalizacion.json"

# Nombres internos de las señales y columnas correspondientes en los CSV
SENALES = ("propofol", "remifentanilo", "bis")
COLUMNAS_CSV = {
    "propofol":      "PPF20_RATE",
    "remifentanilo": "RFTN20_RATE",
    "bis":           "BIS",
}


# =============================================================================
# CÁLCULO, ALMACENAMIENTO Y CARGA DE LAS ESTADÍSTICAS
# =============================================================================
def calcular_estadisticas(archivos_csv, columnas=None):
    """
    Calcula la media y la desviación estándar poblacional de cada señal a
    partir de todas las muestras de los archivos indicados. Debe invocarse
    únicamente con los archivos del conjunto de entrenamiento, a fin de que
    los conjuntos de validación y de prueba no intervengan en el cálculo.

    Parámetros
    ----------
    archivos_csv : lista de rutas a los CSV del conjunto de entrenamiento
    columnas     : diccionario {señal: nombre de columna}; por omisión,
                   COLUMNAS_CSV

    Retorna
    -------
    Diccionario {señal: {"media": float, "desviacion": float, "muestras": int}}
    """
    import pandas as pd

    columnas = COLUMNAS_CSV if columnas is None else columnas
    # Acumuladores de suma, suma de cuadrados y número de muestras, en
    # doble precisión para reducir el error de redondeo
    acumulados = {s: [0.0, 0.0, 0] for s in SENALES}

    for ruta in archivos_csv:
        df = pd.read_csv(ruta, usecols=[columnas[s] for s in SENALES])
        for s in SENALES:
            x = df[columnas[s]].to_numpy(dtype=np.float64)
            x = x[np.isfinite(x)]
            acumulados[s][0] += float(x.sum())
            acumulados[s][1] += float(np.square(x).sum())
            acumulados[s][2] += int(x.size)

    estadisticas = {}
    for s, (suma, suma_cuadrados, n) in acumulados.items():
        if n == 0:
            raise ValueError(f"No se encontraron muestras válidas de la señal '{s}'.")
        media = suma / n
        varianza = max(suma_cuadrados / n - media ** 2, 0.0)
        desviacion = math.sqrt(varianza)
        if desviacion < 1e-12:
            raise ValueError(f"La desviación estándar de la señal '{s}' es nula.")
        estadisticas[s] = {"media": media, "desviacion": desviacion, "muestras": n}
    return estadisticas


def guardar_estadisticas(estadisticas, ruta=RUTA_ESTADISTICAS):
    """Guarda las estadísticas de normalización en un archivo JSON."""
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(estadisticas, archivo, indent=2, ensure_ascii=False)


def cargar_estadisticas(ruta=RUTA_ESTADISTICAS):
    """
    Carga las estadísticas de normalización y verifica que contengan las
    tres señales con una desviación estándar positiva.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de estadísticas: {ruta}. "
            "Debe ejecutarse primero la libreta Crea_tensores_pytorch.ipynb."
        )
    with open(ruta, "r", encoding="utf-8") as archivo:
        estadisticas = json.load(archivo)
    for s in SENALES:
        if s not in estadisticas:
            raise KeyError(f"El archivo de estadísticas no contiene la señal '{s}'.")
        if estadisticas[s]["desviacion"] <= 0:
            raise ValueError(f"La desviación estándar de la señal '{s}' debe ser positiva.")
    return estadisticas


def describir_estadisticas(estadisticas):
    """Devuelve una cadena legible con las estadísticas de normalización."""
    lineas = []
    for s in SENALES:
        e = estadisticas[s]
        lineas.append(f"  {s:14s}: media = {e['media']:.6g}, "
                      f"desviación estándar = {e['desviacion']:.6g}")
    return "\n".join(lineas)


# =============================================================================
# NORMALIZACIÓN
# =============================================================================
def normalizar(x, senal, estadisticas):
    """
    Estandariza la señal indicada con las estadísticas globales. Admite
    arreglos de NumPy y tensores de PyTorch de cualquier forma.
    """
    e = estadisticas[senal]
    return (x - e["media"]) / e["desviacion"]


def desnormalizar(x, senal, estadisticas):
    """Transformación inversa de 'normalizar'."""
    e = estadisticas[senal]
    return x * e["desviacion"] + e["media"]


# =============================================================================
# VENTANAS PARA EL ENTRENAMIENTO Y LA EVALUACIÓN (NumPy)
# =============================================================================
def ventanas_crudas_paciente(propofol, remifentanilo, bis, longitud):
    """
    Construye, en unidades originales, las ventanas de un registro completo.
    La ventana k termina en la muestra k (k = 0, ..., T-1) e incluye el
    relleno descrito en el encabezado del módulo.

    Retorna
    -------
    v_prop, v_remi, v_bis : arreglos (T, longitud) en float32
    """
    propofol = np.asarray(propofol, dtype=np.float32)
    remifentanilo = np.asarray(remifentanilo, dtype=np.float32)
    bis = np.asarray(bis, dtype=np.float32)

    relleno_inf = np.zeros(longitud - 1, dtype=np.float32)
    relleno_bis = np.full(longitud - 1, bis[0], dtype=np.float32)

    ventana = np.lib.stride_tricks.sliding_window_view
    v_prop = ventana(np.concatenate([relleno_inf, propofol]), longitud)
    v_remi = ventana(np.concatenate([relleno_inf, remifentanilo]), longitud)
    v_bis  = ventana(np.concatenate([relleno_bis, bis]), longitud)
    return v_prop, v_remi, v_bis


def construir_ventanas_paciente(propofol, remifentanilo, bis, longitud, estadisticas):
    """
    Genera los pares entrada-etiqueta de un registro para el entrenamiento
    y la evaluación de los modelos. La entrada k abarca las muestras
    k-longitud+1, ..., k y su etiqueta es BIS[k+1], por lo que un registro
    de T muestras produce T-1 pares.

    Retorna
    -------
    Diccionario con:
        infusiones  : (T-1, 2, longitud) float16  — tasas normalizadas
        bis_hist    : (T-1, longitud)    float16  — historial normalizado
        bis_hist_gt : (T-1, longitud)    float32  — historial en unidades de BIS
        etiquetas   : (T-1,)             float32  — BIS en el instante siguiente
    o None si el registro tiene menos de dos muestras.
    """
    bis = np.asarray(bis, dtype=np.float32)
    T = len(bis)
    if T < 2:
        return None

    v_prop, v_remi, v_bis = ventanas_crudas_paciente(propofol, remifentanilo, bis, longitud)
    # Se descarta la última ventana, que carece de valor siguiente
    v_prop, v_remi, v_bis = v_prop[:T - 1], v_remi[:T - 1], v_bis[:T - 1]

    infusiones = np.stack([
        normalizar(v_prop, "propofol", estadisticas),
        normalizar(v_remi, "remifentanilo", estadisticas),
    ], axis=1).astype(np.float16)

    return {
        "infusiones":  infusiones,
        "bis_hist":    normalizar(v_bis, "bis", estadisticas).astype(np.float16),
        "bis_hist_gt": np.ascontiguousarray(v_bis, dtype=np.float32),
        "etiquetas":   bis[1:].astype(np.float32),
    }


# =============================================================================
# VENTANAS PARA LA SIMULACIÓN EN LAZO CERRADO (PyTorch)
# =============================================================================
def construir_ventanas_simulacion(hist_propofol, hist_remifentanilo, hist_bis,
                                  t, longitud, estadisticas, redondeo_float16=True):
    """
    Construye las entradas normalizadas del modelo para un lote de
    simulaciones en el instante t, con el mismo relleno y la misma
    normalización empleados en el entrenamiento.

    Parámetros
    ----------
    hist_propofol, hist_remifentanilo : (N, T_total) tasas aplicadas a la
                                        planta, en mL/s
    hist_bis         : (N, T_total) BIS simulado; la columna 0 corresponde a
                       la condición inicial y se emplea como relleno
    t                : instante en que termina la ventana (inclusive)
    longitud         : longitud de la ventana del modelo
    estadisticas     : estadísticas globales de normalización
    redondeo_float16 : si es True, las entradas se redondean a float16,
                       precisión con la que se almacenaron los tensores de
                       entrenamiento

    Retorna
    -------
    infusiones : (N, 2, longitud)
    bis_hist   : (N, longitud)
    """
    if torch is None:
        raise ImportError("Se requiere PyTorch para construir las ventanas de simulación.")

    inicio = max(0, t - longitud + 1)
    n_relleno = longitud - (t + 1 - inicio)

    v_p = hist_propofol[:, inicio:t + 1]
    v_r = hist_remifentanilo[:, inicio:t + 1]
    v_b = hist_bis[:, inicio:t + 1]

    if n_relleno > 0:
        N = v_p.shape[0]
        ceros = v_p.new_zeros(N, n_relleno)
        v_p = torch.cat([ceros, v_p], dim=1)
        v_r = torch.cat([ceros, v_r], dim=1)
        v_b = torch.cat([hist_bis[:, :1].expand(N, n_relleno), v_b], dim=1)

    infusiones = torch.stack([
        normalizar(v_p, "propofol", estadisticas),
        normalizar(v_r, "remifentanilo", estadisticas),
    ], dim=1)
    bis_hist = normalizar(v_b, "bis", estadisticas)

    if redondeo_float16:
        infusiones = infusiones.half().float()
        bis_hist = bis_hist.half().float()
    return infusiones, bis_hist


def ventanas_infusion_tensor(infusion, longitud, estadisticas, redondeo_float16=True):
    """
    Genera todas las ventanas normalizadas de un par de señales de infusión
    (útil para protocolos sintéticos). La ventana k termina en el paso k y
    los instantes previos al inicio se rellenan con cero.

    Parámetros
    ----------
    infusion : (2, N) tasas de propofol y de remifentanilo en mL/s

    Retorna
    -------
    (N, 2, longitud)
    """
    if torch is None:
        raise ImportError("Se requiere PyTorch para construir las ventanas.")
    infusion = torch.as_tensor(infusion, dtype=torch.float32)
    relleno = torch.zeros(infusion.shape[0], longitud - 1)
    extendida = torch.cat([relleno, infusion], dim=1)
    crudas = extendida.unfold(1, longitud, 1).permute(1, 0, 2)      # (N, 2, longitud)
    salida = torch.stack([
        normalizar(crudas[:, 0], "propofol", estadisticas),
        normalizar(crudas[:, 1], "remifentanilo", estadisticas),
    ], dim=1)
    return salida.half().float() if redondeo_float16 else salida


def normalizar_bis_tensor(bis_crudo, estadisticas, redondeo_float16=True):
    """Normaliza un tensor de BIS en unidades originales con las estadísticas globales."""
    salida = normalizar(bis_crudo.float(), "bis", estadisticas)
    return salida.half().float() if redondeo_float16 else salida
