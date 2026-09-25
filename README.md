# Anestesia e Inteligencia Artificial: Modelado y Regulación del Nivel de Hipnosis

Repositorio de código asociado a la tesis de posgrado **"Anestesia e Inteligencia Artificial: Modelado y Regulación del Nivel de Hipnosis Durante la Infusión Controlada por Objetivo"**, desarrollada en el Posgrado en Ciencia e Ingeniería de la Computación del IIMAS, UNAM.

En este repositorio se reúne el flujo completo de trabajo empleado para predecir el Índice Biespectral (BIS) a partir de las infusiones de propofol y remifentanilo, así como para sintonizar y estimar controladores PID que regulan el nivel de hipnosis en lazo cerrado.

---

## Descripción general

El trabajo se organiza en cuatro etapas:

1. **Adquisición y preparación de datos.** Se extraen registros intraoperatorios de la base de datos abierta [VitalDB](https://vitaldb.net). Se conservan únicamente los casos con señales de BIS, propofol y remifentanilo de calidad suficiente. Después, las señales se limpian, se remuestrean y se convierten en tensores de PyTorch.
2. **Modelado del BIS.** Se entrenan y comparan tres predictores del BIS: una red LSTM de referencia, un modelo híbrido (CNN 1D + LSTM + GRN + atención multicabeza) inspirado en la estructura farmacocinética-farmacodinámica (PK-PD), y un modelo PK-PD clásico basado en ecuaciones diferenciales.
3. **Regulación del nivel de hipnosis.** El predictor del BIS se emplea como planta virtual. Sobre ella se sintonizan, mediante un algoritmo genético, un PID proporcional y un PID dual. Las señales de control se filtran con una media exponencial móvil (EMA) antes de aplicarse a la planta.
4. **Estimación de hiperparámetros del controlador.** Se propone una heurística basada en K vecinos más cercanos (KNN) que estima las ganancias del PID de un paciente nuevo a partir de sus covariables (edad, peso, altura y sexo). La relación entre covariables y ganancias se analiza además mediante componentes principales (PCA).

---

## Estructura del repositorio

| Archivo | Etapa | Descripción |
|---|---|---|
| `Adquirir_Datos.ipynb` | 1 | Filtrado de calidad de los casos de VitalDB: porcentaje de datos faltantes en los volúmenes de infusión y huecos de 30 s o más en el BIS. Recorte sincronizado de las señales y descarga de los datos demográficos. |
| `Procesar_y_limpiar_datos.ipynb` | 1 | Limpieza de las señales: valores de BIS inferiores a 20 y volúmenes acumulados decrecientes se sustituyen por interpolación lineal. Remuestreo en bloques de 10 s (tasas de infusión y BIS promedio), suavizado LOWESS (solo en el conjunto de entrenamiento) y normalización min-max de las covariables. |
| `estats_covariables.ipynb` | 1 | Estadística descriptiva de la población (640 pacientes: 357 hombres y 283 mujeres), total y estratificada por sexo. |
| `Crea_tensores_pytorch.ipynb` | 1 | Generación de ventanas deslizantes de 180 muestras con relleno de ceros y normalización por ventana. Los conjuntos de entrenamiento, validación y prueba se exportan como tensores `.pt`. |
| `Modelo_LSTM_BIS.py` | 2 | Definición de `ModeloLSTMBIS`: dos LSTM independientes (una por fármaco) y un regresor MLP. |
| `Modelo_LSTM_BIS_SLD.ipynb` | 2 | Entrenamiento de la LSTM de referencia con una pérdida SLD (*Smooth Label Distribution*), que pondera cada muestra de forma inversa a la densidad de etiquetas. |
| `Modelo_Predictor_BIS.py` | 2 | Definición de `ModeloAnestesia`, el predictor híbrido compuesto por tres bloques (véase la sección de modelos). |
| `Modelo_Predictor_BIS.ipynb` | 2 | Entrenamiento del predictor híbrido con dos pérdidas: MSE auxiliar sobre el BIS pseudo-histórico y MSE ponderada por densidad de etiquetas (LDS) sobre la predicción final. |
| `Comparacion_Modelos_BIS.ipynb` | 2 | Comparación de los tres predictores mediante MAE, MSE, RMSE y MAPE. El modelo PK-PD emplea los modelos de Schnider (propofol) y Minto (remifentanilo) con una superficie de interacción tipo Hill (Ionescu *et al.*, IEEE Access, 2021). |
| `PID_Filtro_EMA.ipynb` | 3 | Sintonización por algoritmo genético del PID proporcional (`Kp, Ki, Kd, γ`) y del PID dual (`Kp, Ki, Kd` para cada fármaco), con filtrado EMA. Incluye la sintonización poblacional y la comparación estadística entre ambos controladores (Shapiro-Wilk y Wilcoxon). |
| `Graficas_Sintonizaciones_PID.ipynb` | 3 | Reproducción de la respuesta en lazo cerrado de cada paciente con las ganancias sintonizadas, sobre la trayectoria clínica de sedación, mantenimiento y recuperación. |
| `KNN_Estimacion_PID.ipynb` | 4 | Estimación de las ganancias del PID proporcional mediante KNN ponderado con distancia de Minkowski y pesos *softmax*. Se evalúan tres variantes: población global, solo hombres y solo mujeres. |
| `PCA_Respuesta_de_Control.mlx` | 4 | Script en vivo de MATLAB con el análisis de componentes principales de las covariables y las constantes del controlador. |
| `Poblacion_Prueba.csv` | 3–4 | Población de 340 pacientes (edad, peso, altura y sexo, en valores naturales y normalizados) empleada para la sintonización. |
| `SINTONIZACIONES.csv` | 3–4 | Ganancias del PID proporcional (`Kp, Ki, Kd, γ`) y RMSE obtenidos para cada paciente de la población. |
| `Sinto.csv` | 3 | Ganancias del PID dual y RMSE obtenidos para un subconjunto de 100 pacientes. |

---

## Modelos de predicción del BIS

### `ModeloAnestesia` (predictor híbrido)

- **Bloque A (estimador del BIS pseudo-histórico).** Una CNN 1D "cinética" estima una concentración en el sitio efector a partir de las infusiones. Esa concentración se refina con una *Gated Residual Network* (GRN) condicionada por las covariables. Una CNN 1D "dinámica" estima, a partir del historial del BIS, los parámetros de una curva sigmoidal:

  $$\mathrm{BIS}_k = W_0 + W_1\,\sigma\!\left(W_2\,C_{e,k} + W_3\right)$$

- **Bloque B.** Tres LSTM independientes procesan el propofol, el remifentanilo y el BIS pseudo-histórico. Sus salidas se fusionan mediante GRN, con las covariables del paciente como contexto.
- **Bloque C.** Una atención multicabeza interpretable y un *bottleneck* final producen la predicción del BIS en el instante siguiente.

### `ModeloLSTMBIS` (referencia)

Consta de dos LSTM apiladas e independientes para cada fármaco, cuyos estados ocultos finales se concatenan y alimentan un regresor MLP.

---

## Esquema de control

- **Planta:** el predictor `ModeloAnestesia` previamente entrenado.
- **PID proporcional:** un solo PID calcula la infusión de propofol, y la de remifentanilo se obtiene como `u_remi = γ · u_ppf`.
- **PID dual:** se emplean dos PID independientes, uno para cada fármaco.
- **Filtrado:** a las señales de control se les aplica una media exponencial móvil sobre una ventana deslizante de longitud fija, `y[n] = α·u[n] + (1 − α)·y[n−1]`.
- **Trayectoria de referencia Λ(t):**

| Fase | Intervalo | Referencia |
|---|---|---|
| Sedación | 0 – 300 s | `Λ(t) = 98 + 56.8·tanh(−0.004·t)` |
| Mantenimiento | 300 – 3900 s | `Λ(t) = 50` |
| Recuperación | 3900 – 4440 s | `Λ(τ) = 50 + 49.7·tanh(0.004·τ)` |

- **Índice de desempeño para la estimación KNN:**

$$\xi = \frac{2}{T_f^{2}}\int_{0}^{T_f} t\,\left|\frac{e(t)}{\Lambda(t)}\right|dt, \qquad e(t) = \mathrm{BIS}(t) - \Lambda(t)$$

---

## Resultados principales

**Predicción del BIS** (media ± desviación estándar sobre los registros de evaluación):

| Modelo | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| LSTM (SLD) | 9.468 ± 3.354 | 11.747 ± 3.547 | 24.40 ± 13.95 |
| **ModeloAnestesia** | **8.235 ± 2.551** | **10.405 ± 2.748** | **20.45 ± 9.48** |
| PK-PD (ecuaciones diferenciales) | 18.693 ± 10.261 | 21.796 ± 9.774 | 47.39 ± 37.93 |

**Control:** en 18 pacientes, el PID proporcional (RMSE = 5.38 ± 1.57) y el PID dual (RMSE = 5.38 ± 1.65) no presentaron una diferencia estadísticamente significativa (prueba de Wilcoxon, p = 0.52).

**Estimación KNN** (K = 2):

| Variante | Pacientes estimados | Exponente de Minkowski óptimo | ξ medio ± desv. est. |
|---|---|---|---|
| Global | 174 | 1.4 | 0.163 ± 0.064 |
| Hombres | 76 | 1.1 | 0.169 ± 0.041 |
| Mujeres | 98 | 1.5 | 0.155 ± 0.078 |

---

## Requisitos

- Python 3.13
- [PyTorch](https://pytorch.org) (se recomienda una GPU con CUDA)
- `vitaldb`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`
- MATLAB (solo para `PCA_Respuesta_de_Control.mlx`)

```bash
pip install torch vitaldb numpy pandas scipy statsmodels matplotlib seaborn
```

---

## Orden de ejecución

1. `Adquirir_Datos.ipynb`
2. `Procesar_y_limpiar_datos.ipynb`
3. `estats_covariables.ipynb` (opcional)
4. `Crea_tensores_pytorch.ipynb`
5. `Modelo_LSTM_BIS_SLD.ipynb` y `Modelo_Predictor_BIS.ipynb`
6. `Comparacion_Modelos_BIS.ipynb`
7. `PID_Filtro_EMA.ipynb`
8. `Graficas_Sintonizaciones_PID.ipynb`
9. `KNN_Estimacion_PID.ipynb`
10. `PCA_Respuesta_de_Control.mlx`

> **Nota:** las rutas de entrada y salida están definidas como rutas absolutas en la sección de configuración de cada libreta. Antes de ejecutarlas, dichas rutas deben ajustarse al entorno local.

---

## Datos

Los registros clínicos provienen de **VitalDB**, una base de datos abierta de señales vitales intraoperatorias. 

> Lee H-C, Park Y, Yoon SB, *et al.* VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Scientific Data*, 9, 279 (2022).

---

## Autoría

- **Autor:** Ing. Yordi Castellanos Márquez
- **Director de tesis:** Dr. Luis Álvarez-Icaza Longoria
- **Institución:** Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS), Universidad Nacional Autónoma de México (UNAM)
