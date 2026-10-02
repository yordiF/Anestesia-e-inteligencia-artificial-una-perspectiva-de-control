# Anestesia e Inteligencia Artificial: Modelado y Regulación del Nivel de Hipnosis

Repositorio de código asociado a la tesis de posgrado **«Anestesia e Inteligencia Artificial: Modelado y Regulación del Nivel de Hipnosis Durante la Infusión Controlada por Objetivo»**, desarrollada en el Posgrado en Ciencia e Ingeniería de la Computación del Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS) de la Universidad Nacional Autónoma de México (UNAM).

En este repositorio se reúne el flujo completo de trabajo empleado para predecir el Índice Biespectral (BIS) a partir de las infusiones de propofol y remifentanilo, para sintonizar y estimar controladores PID que regulan el nivel de hipnosis en lazo cerrado, y para analizar estadísticamente los resultados reportados en la tesis.

---

## Descripción general

El trabajo se organiza en cinco etapas:

1. **Adquisición y preparación de los datos.** Se extraen registros intraoperatorios de la base de datos abierta [VitalDB](https://vitaldb.net) correspondientes a procedimientos bajo anestesia total intravenosa. Se conservan únicamente los casos con señales de BIS, propofol y remifentanilo de calidad suficiente y con datos demográficos completos. Posteriormente, las señales se limpian, se remuestrean y se convierten en tensores de PyTorch.
2. **Modelado del BIS.** Se entrenan y se comparan tres predictores del BIS: una red LSTM de referencia, un modelo híbrido (CNN 1D + LSTM + GRN + atención multicabeza) inspirado en la estructura farmacocinética-farmacodinámica (PK-PD) y un modelo PK-PD clásico basado en ecuaciones diferenciales.
3. **Regulación del nivel de hipnosis.** El predictor del BIS se emplea como planta virtual. Sobre ella se sintonizan, mediante un algoritmo genético, dos arquitecturas de control: un PID proporcional (Arquitectura I) y un PID dual (Arquitectura II). Las señales de control se filtran mediante una media móvil exponencial (EMA) antes de aplicarse a la planta.
4. **Estimación de hiperparámetros del controlador.** Se propone una heurística basada en los K vecinos más cercanos (KNN) que estima las ganancias del PID de un paciente nuevo a partir de sus covariables (edad, peso, altura y sexo). La relación entre las covariables y las ganancias se analiza, además, mediante componentes principales (PCA).
5. **Análisis estadístico complementario y figuras de la tesis.** Se comparan formalmente los modelos predictores, las arquitecturas de control y la heurística de estimación, y se cuantifica el desempeño de la heurística frente a otras estrategias de parametrización.

---

## Estructura del repositorio

### Libretas y scripts

| Archivo | Etapa | Descripción |
|---|---|---|
| `Adquirir_Datos.ipynb` | 1 | Filtrado de calidad de los casos de VitalDB: porcentaje de datos faltantes en los volúmenes de infusión y huecos de 30 s o más en el BIS. Recorte sincronizado de las señales y descarga de los datos demográficos. |
| `Procesar_y_limpiar_datos.ipynb` | 1 | Limpieza de las señales: los valores de BIS inferiores a 20 y los volúmenes acumulados decrecientes se sustituyen por interpolación lineal. Remuestreo en bloques de 10 s (tasas de infusión y BIS promedio), suavizado LOWESS (únicamente en el conjunto de entrenamiento) y normalización mínimo-máximo de las covariables. |
| `estats_covariables.ipynb` | 1 | Estadística descriptiva de la población (640 pacientes: 357 hombres y 283 mujeres), total y estratificada por sexo. |
| `Normalizacion_BIS.py` | 1 | Funciones comunes para construir y normalizar las ventanas de entrada de los predictores del BIS: normalización global con la media y la desviación estándar del conjunto de entrenamiento, relleno con cero en las infusiones y con el primer valor en el BIS, y alineación de cada ventana, que termina en el instante *n*, con la etiqueta BIS[*n*+1]. Lo emplean todas las libretas que entrenan, evalúan o simulan los modelos. |
| `Crea_tensores_pytorch.ipynb` | 1 | Cálculo de las estadísticas globales de normalización sobre el conjunto de entrenamiento (`estadisticas_normalizacion.json`) y generación de las ventanas deslizantes de 180 muestras mediante `Normalizacion_BIS.py`. Los conjuntos de entrenamiento, validación y prueba se exportan como tensores `.pt`. |
| `Modelo_LSTM_BIS.py` | 2 | Definición de `ModeloLSTMBIS`: dos redes LSTM independientes (una por fármaco) y un regresor MLP. |
| `Modelo_LSTM_BIS_SLD.ipynb` | 2 | Entrenamiento de la LSTM de referencia con una pérdida ponderada por la densidad de las etiquetas (LDS, *Label Distribution Smoothing*; denominada SLD en la libreta). |
| `Modelo_Predictor_BIS.py` | 2 | Definición de `ModeloAnestesia`, el predictor híbrido compuesto por tres bloques (véase la sección de modelos). |
| `Modelo_Predictor_BIS.ipynb` | 2 | Entrenamiento del predictor híbrido con dos pérdidas: una MSE auxiliar sobre el BIS pseudo-histórico y una MSE ponderada por densidad de etiquetas (LDS) sobre la predicción final. |
| `Comparacion_Modelos_BIS.ipynb` | 2 | Comparación de los tres predictores mediante MAE, MSE, RMSE y MAPE. El modelo PK-PD emplea los modelos de Schnider (propofol) y de Minto (remifentanilo) con una superficie de interacción de tipo Hill (Ionescu *et al.*, *IEEE Access*, 2021). |
| `Atencion_Promedio_Interpretacion_Clinica.ipynb` | 2 | Interpretación clínica de la atención promedio del Bloque C de `ModeloAnestesia` ante un protocolo de 130 min con trenes de pulsos de propofol y remifentanilo: horizonte de la atención, atención relativa sobre el bolo y sobre el periodo posterior, y robustez ante distintos contextos de paciente. |
| `PID_Filtro_EMA.ipynb` | 3 | Sintonización mediante algoritmo genético del PID proporcional (`Kp, Ki, Kd, γ`) y del PID dual (`Kp, Ki, Kd` para cada fármaco), con filtrado EMA. Incluye los experimentos E1 a E5 y la comparación estadística inicial entre ambos controladores (Shapiro–Wilk y Wilcoxon). |
| `Graficas_Sintonizaciones_PID.ipynb` | 3 | Reproducción de la respuesta en lazo cerrado de cada paciente con las ganancias sintonizadas, sobre la trayectoria clínica de sedación, mantenimiento y recuperación. |
| `KNN_Estimacion_PID.ipynb` | 4 | Estimación de las ganancias del PID proporcional mediante KNN ponderado con la distancia de Minkowski y pesos *softmax*. Se evalúan tres variantes: población global, únicamente hombres y únicamente mujeres. |
| `Grafica_Superficie_KP_KD_rho.ipynb` | 4 | Representación de las sintonizaciones del AG como una superficie sobre el plano (`Kp`, `Kd`), cuya altura corresponde a ρ y cuyo color codifica el RMSE, obtenida mediante regresión por núcleo gaussiano (Nadaraya–Watson); incluye mapas 2D y proyecciones por pares. |
| `PCA_Respuesta_de_Control.mlx` | 4 | Script en vivo de MATLAB con el análisis de componentes principales de las covariables y de las constantes del controlador. |
| `Analisis_Estadistico_Modelos_BIS.ipynb` | 5 | Prueba de Friedman y pruebas de rangos con signo de Wilcoxon con corrección de Holm entre los tres predictores del BIS (Tablas 5.1 y 5.2 de la tesis). |
| `Analisis_Estadistico_Arquitecturas_E3.ipynb` | 5 | Comparación entre las Arquitecturas I y II del experimento E3: Shapiro–Wilk, Wilcoxon, prueba de equivalencia TOST con análisis de sensibilidad del margen δ, estimador de Hodges–Lehmann, correlación rango-biserial, intervalo *bootstrap* y efecto del sexo (Tablas 5.4 y 5.5). |
| `Estadisticas_Base_Sintonizaciones.ipynb` | 5 | Descripción de la base de 340 sintonizaciones: RMSE por sexo, estadística de los parámetros, parámetros en los límites del intervalo de búsqueda y correlaciones de Spearman con las covariables (Sección 5.5 y Tabla 5.7). |
| `Comparacion_KNN_AG_Mediana.ipynb` | 5 | Simulación, con la misma planta virtual, de las ganancias del algoritmo genético, de las estimadas por la heurística KNN y de un controlador de parámetros medianos, en el conjunto de estimación y en los casos con sintonización deficiente; incluye el tiempo de cómputo de la heurística y las pruebas de Wilcoxon (Tablas 5.10 y 5.11). |

### Archivos de datos

| Archivo | Origen | Descripción |
|---|---|---|
| `Poblacion_Prueba.csv` | — | Población de 340 pacientes virtuales (edad, peso, altura y sexo, en valores naturales y normalizados), obtenida como una muestra aleatoria de pacientes cuyos datos no se emplearon en el entrenamiento del modelo predictor. |
| `SINTONIZACIONES.csv` | `PID_Filtro_EMA.ipynb` (E5) | Ganancias del PID proporcional (`Kp, Ki, Kd, γ`) y RMSE obtenidos para cada paciente de la población. |
| `Sinto.csv` | `PID_Filtro_EMA.ipynb` | Ganancias del PID dual y RMSE obtenidos para un subconjunto de 100 pacientes. |
| `Metricas_Modelos_por_Paciente.csv` | `Comparacion_Modelos_BIS.ipynb` | MAE, MSE, RMSE y MAPE de cada modelo para cada uno de los 32 pacientes del conjunto de prueba. |
| `Perdidas_Entrenamiento_Modelos.csv` | `Modelo_Predictor_BIS.ipynb` y `Modelo_LSTM_BIS_SLD.ipynb` | Pérdida LDS de entrenamiento y de validación por época de ambos modelos de aprendizaje profundo. |
| `RMSE_Arquitecturas_E3.csv` | `PID_Filtro_EMA.ipynb` (E3) | Covariables, RMSE y parámetros sintonizados de ambas arquitecturas para los 18 pacientes del experimento E3. |
| `Barrido_p_KNN.csv` | `KNN_Estimacion_PID.ipynb` | Valor de ξ y error absoluto medio de los parámetros para cada orden p de la distancia de Minkowski en las tres variantes de la heurística. |
| `Resultados_Comparacion_KNN_AG_Mediana.csv` y `Trayectorias_BIS_Comparacion.npz` | `Comparacion_KNN_AG_Mediana.ipynb` | Métricas por paciente y trayectorias del BIS simuladas para las tres estrategias de parametrización. Se generan al ejecutar la libreta. |

Los archivos cuyo origen es una libreta de las etapas 2 a 4 se construyeron a partir de las salidas registradas en dicha libreta, con el fin de que los análisis de la etapa 5 puedan ejecutarse sin repetir los entrenamientos ni las sintonizaciones.

---

## Modelos de predicción del BIS

### `ModeloAnestesia` (predictor híbrido)

- **Bloque A (estimador del BIS pseudo-histórico).** Una CNN 1D «cinética» estima una concentración en el sitio de efecto a partir de las infusiones. Dicha concentración se refina mediante una *Gated Residual Network* (GRN) condicionada por las covariables. Una CNN 1D «dinámica» estima, a partir del historial del BIS, los parámetros de una curva sigmoidal:

  $$\mathrm{BIS}^{\mathrm{ps}}[n] = \psi_0 + \psi_1\,\mathrm{sig}\!\left(\psi_2\,\hat{C}_e[n] + \psi_3\right)$$

- **Bloque B.** Tres redes LSTM independientes procesan el propofol, el remifentanilo y el BIS pseudo-histórico. Sus salidas se fusionan mediante redes GRN, con las covariables del paciente como contexto.
- **Bloque C.** Una atención multicabeza interpretable y una capa de compresión (*bottleneck*) final producen la predicción del BIS en el instante siguiente.

### `ModeloLSTMBIS` (referencia)

Consta de dos redes LSTM apiladas e independientes, una para cada fármaco, cuyos estados ocultos finales se concatenan y alimentan un regresor MLP.

---

## Esquema de control

- **Planta:** el predictor `ModeloAnestesia` previamente entrenado.
- **PID proporcional (Arquitectura I):** un único PID calcula la infusión de propofol, y la de remifentanilo se obtiene como `u_remi = γ · u_ppf`. En la tesis, el factor de escalamiento γ se denota como ρ.
- **PID dual (Arquitectura II):** se emplean dos PID independientes, uno para cada fármaco.
- **Filtrado:** a las señales de control se les aplica una media móvil exponencial sobre una ventana deslizante de longitud fija, `y[n] = α·u[n] + (1 − α)·y[n−1]`, con ventana de 12 muestras y α = 0.05.
- **Saturación:** la señal cruda de cada controlador se acota a `U_MAX = 0.05` mL/s, de modo que la infusión filtrada permanece dentro del intervalo de tasas observado en el conjunto de entrenamiento del predictor.
- **Trayectoria de referencia Λ(t):**

| Fase | Intervalo | Referencia |
|---|---|---|
| Sedación | 0 – 300 s | `Λ(t) = 98 + 56.8·tanh(−0.004·t)` |
| Mantenimiento | 300 – 3900 s | `Λ(t) = 50` |
| Recuperación (lazo abierto) | 3900 – 5700 s | `Λ(τ) = 50 + 42·tanh(0.003·τ)` |

- **Recuperación en lazo abierto:** durante los 30 minutos de la fase de recuperación el PID se desactiva y la planta evoluciona sin control; en esta fase Λ(t) se conserva únicamente como curva de comparación.
- **Función de costo de la sintonización poblacional (E4 y E5)**, ITAE normalizado y evaluado mientras el controlador permanece activo, donde `T_c` es la duración conjunta de la sedación y el mantenimiento:

$$J = \frac{2}{T_c^{2}}\int_{0}^{T_c} t\,|e(t)|\,dt$$

- **Función de error para la evaluación de la heurística KNN**, sobre el mismo horizonte:

$$\xi = \frac{2}{T_c^{2}}\int_{0}^{T_c} t\,\left|\frac{e(t)}{\Lambda(t)}\right|dt, \qquad e(t) = \mathrm{BIS}(t) - \Lambda(t)$$

- **Casos con sintonización deficiente:** pacientes cuyo RMSE constituye un valor atípico superior según el criterio de Tukey (`RMSE > Q3 + 1.5·RIC`).

---

## Resultados principales

**Predicción del BIS** (media ± desviación estándar sobre los 32 pacientes del conjunto de prueba):

| Modelo | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| LSTM (LDS) | 9.213 ± 7.009 | 11.329 ± 6.757 | 23.18 ± 24.44 |
| **ModeloAnestesia** | **6.301 ± 1.407** | **8.133 ± 1.905** | **15.70 ± 6.54** |
| PK-PD (ecuaciones diferenciales) | 18.693 ± 10.261 | 21.796 ± 9.774 | 47.39 ± 37.93 |

Las diferencias entre los tres modelos son estadísticamente significativas (prueba de Friedman, valor-p < 10⁻¹⁰; pruebas de Wilcoxon con corrección de Holm).

**Control:** en los 18 pacientes del experimento E3 (referencia constante de 40 minutos, con inducción desde BIS = 98), el PID proporcional (RMSE = 11.63 ± 0.12) y el PID dual (RMSE = 11.62 ± 0.13) no presentaron una diferencia estadísticamente significativa (prueba de Wilcoxon, valor-p = 0.67), y su equivalencia se demostró mediante la prueba TOST con un margen δ = 0.5 unidades de BIS (valor-p < 10⁻⁴).

**Base de sintonizaciones (E5):** RMSE en sedación y mantenimiento de 5.60 ± 0.26 en los 340 pacientes; 6 casos atípicos según el criterio de Tukey (umbral 6.22).

**Estimación KNN** (K = 2; Espacio de 134 pacientes, 67 hombres y 67 mujeres):

| Variante | Pacientes estimados | Exponente de Minkowski óptimo | ξ medio ± desv. est. |
|---|---|---|---|
| Global | 200 | 1.0 | 0.0162 ± 0.0029 |
| Hombres | 100 | 2.2 | 0.0167 ± 0.0030 |
| Mujeres | 100 | 1.0 | 0.0157 ± 0.0020 |

**Comparación de estrategias** (conjunto de estimación, 200 pacientes): ξ = 0.0173 ± 0.0018 con las ganancias del AG, 0.0162 ± 0.0029 con la heurística KNN y 0.0147 ± 0.0021 con el controlador de parámetros medianos del Espacio.

---

## Requisitos

- Python 3.13
- [PyTorch](https://pytorch.org) (se recomienda una GPU con soporte para CUDA)
- `vitaldb`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`
- MATLAB (únicamente para `PCA_Respuesta_de_Control.mlx`)

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
6. `Comparacion_Modelos_BIS.ipynb` y `Atencion_Promedio_Interpretacion_Clinica.ipynb`
7. `PID_Filtro_EMA.ipynb`
8. `Graficas_Sintonizaciones_PID.ipynb`
9. `KNN_Estimacion_PID.ipynb`
10. `PCA_Respuesta_de_Control.mlx` y `Grafica_Superficie_KP_KD_rho.ipynb`
11. `Analisis_Estadistico_Modelos_BIS.ipynb`, `Analisis_Estadistico_Arquitecturas_E3.ipynb` y `Estadisticas_Base_Sintonizaciones.ipynb` (independientes entre sí)
12. `Comparacion_KNN_AG_Mediana.ipynb`

> **Nota:** en varias libretas, las rutas de entrada y de salida se definen como rutas absolutas en la sección de configuración. Antes de ejecutarlas, dichas rutas deben ajustarse al entorno local. Las libretas del paso 11 emplean únicamente los archivos de datos descritos en la sección «Archivos de datos».

---

## Datos

Los registros clínicos provienen de **VitalDB**, una base de datos abierta de señales vitales intraoperatorias:

> Lee H-C, Park Y, Yoon SB, *et al.* VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Scientific Data*, 9, 279 (2022).

---

## Autoría

- **Autor:** Ing. Yordi Castellanos Márquez
- **Director de tesis:** Dr. Luis Álvarez-Icaza Longoria
- **Institución:** Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS), Universidad Nacional Autónoma de México (UNAM)
