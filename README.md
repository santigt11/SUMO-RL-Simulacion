# Comparación de Q-Learning y DQN para Control Semafórico en Loja

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Estructura del Directorio](#2-estructura-del-directorio)
3. [Requisitos Previos](#3-requisitos-previos)
4. [Pipeline de Datos](#4-pipeline-de-datos)
5. [Simulación de Línea Base (Tiempos Fijos)](#5-simulación-de-línea-base-tiempos-fijos)
6. [Q-Learning Tabular](#6-q-learning-tabular)
7. [Deep Q-Network (DQN)](#7-deep-q-network-dqn)
8. [Clases Principales](#8-clases-principales)
9. [Dashboard](#9-dashboard)
10. [Archivos de Salida por Escenario](#10-archivos-de-salida-por-escenario)
11. [Formato de Datos](#11-formato-de-datos)
12. [Reproducibilidad](#12-reproducibilidad)
13. [Referencias](#13-referencias)

## 1. Descripción del Proyecto

Este proyecto compara tres estrategias de control semafórico aplicadas a una intersección de la ciudad de Loja (Av. Isidro Ayora x 8 de Diciembre):

1. **Tiempos Fijos (Baseline):** Controlador convencional con ciclos predefinidos de 120 segundos.
2. **Q-Learning Tabular:** Agente de aprendizaje por refuerzo que discretiza el estado de tráfico en 288 estados y 12 acciones.
3. **Deep Q-Network (DQN):** Agente de aprendizaje profundo que procesa el estado continuo mediante una red neuronal MLP [128, 128].

Los tres escenarios utilizan la misma demanda vehicular generada estocásticamente (distribución de Weibull, 2334 vehículos en 3600 segundos) y la misma función de recompensa multi-objetivo.

---

## 2. Estructura del Directorio

```
Tesis/
├── Simple_Intersection.net.xml          # Red de SUMO (netedit 1.27.0)
├── Distribuciones/
│   ├── distribucion_weibull.ipynb       # Generación de demanda estocástica
│   ├── validacion_entorno.ipynb         # Validación de archivos SUMO
│   └── loja_intersection_weibull.rou.xml # Archivo de rutas generado
├── Simulaciones_tiempos_fijos/
│   ├── baseline.ipynb                   # Simulación de línea base
│   └── resultados/
│       ├── metricas.csv                 # Métricas temporales
│       ├── tripinfo.xml                 # Info de viajes por vehículo
│       └── bin_edges_calibrados.npz     # Límites de discretización
├── Q-Learning/
│   ├── qlearning.ipynb                  # Entrenamiento Q-Learning
│   └── resultados/
│       ├── qlearning.pkl                # Tabla Q serializada
│       ├── metricas.csv
│       └── tripinfo.xml
├── DQN/
│   ├── dqn.ipynb                        # Entrenamiento DQN
│   └── resultados/
│       ├── best_model.zip               # Mejor modelo DQN
│       ├── dqn_ultimo.zip               # Último modelo DQN
│       ├── metricas.csv
│       └── tripinfo.xml
├── Tratamiento_Datos/
│   └── procesamiento_aforos.ipynb       # Procesamiento de conteos vehiculares
├── dashboard/
│   ├── app.py                           # Aplicación Streamlit
│   ├── utils.py                         # Funciones de carga de datos
│   ├── run.sh                           # Script de inicio
│   ├── requirements.txt
│   └── README.md
├── .venv/                               # Entorno virtual Python 3.14.5
└── .gitignore
```

---

## 3. Requisitos Previos

### 3.1 Software

- **Python 3.14.5** (o superior)
- **SUMO 1.26.0** o superior (con `SUMO_HOME` configurado)
- **Git** (para clonar el repositorio)

### 3.2 Dependencias de Python

Las dependencias están gestionadas en `/home/santiago/Documentos/Tesis/.venv/`. Paquetes principales:

| Paquete | Versión | Uso |
|---|---|---|
| sumo-rl | 1.2.x | Entorno de simulación para RL |
| stable-baselines3 | 2.x | Implementación de DQN |
| gymnasium | 0.29.x | Framework de entornos RL |
| pandas | 2.1.x | Manipulación de datos |
| numpy | 1.26.x | Cálculo numérico |
| torch | 2.x | Redes neuronales (DQN) |
| streamlit | 1.59.x | Dashboard web |
| plotly | 5.x | Gráficos interactivos |
| lxml | 5.x | Parsing de XML |

Para instalar las dependencias del dashboard:

```bash
/home/santiago/Documentos/Tesis/.venv/bin/pip install streamlit plotly pandas lxml
```

---

## 4. Pipeline de Datos

### 4.1 Procesamiento de Aforos Vehiculares

**Archivo:** `Tratamiento_Datos/procesamiento_aforos.ipynb`

Este notebook procesa los conteos empíricos de vehículos realizados a partir de grabaciones de video en la intersección (datos de la tesis de Sarango [2025]). Los pasos son:

1. Lectura de registros de conteo vehicular por tipo (car, moto, bus).
2. Cálculo de tasas de flujo (veh/s, veh/min, veh/h).
3. Clasificación por acceso (Norte vía -E5, Oeste vía -E4).
4. Descarga del dataset de imágenes etiquetadas desde Roboflow (3 clases: car, moto, bus).

**Resultado:** Distribución por tipo de vehículo: 74.1% car, 14.8% moto, 11.1% bus. Flujo promedio: 1166.9 veh/hora por acceso.

### 4.2 Generación de Demanda Estocástica

**Archivo:** `Distribuciones/distribucion_weibull.ipynb`

Genera el archivo de rutas `loja_intersection_weibull.rou.xml` utilizando una distribución de Weibull para modelar la variabilidad temporal de las llegadas.

**Función de densidad de probabilidad:**

$$\lambda(t) = \frac{k}{\lambda_{w}} \left( \frac{t}{\lambda_{w}} \right)^{k-1} e^{-\left( \frac{t}{\lambda_{w}} \right)^k}$$

**Parámetros:**

- k = 2.5 (forma): Controla la asimetría de la curva.
- λ_w = 0.55 × 3600 (escala): Localiza el pico entre 20-30 minutos.
- Total de vehículos: 2334.
- Duración: 3600 segundos.
- Seed: 42 (reproducibilidad).

**Rutas generadas (4 movimientos):**

- Norte → Sur (directo)
- Norte → Este (giro izquierdo)
- Oeste → Este (directo)
- Oeste → Sur (giro derecho)

### 4.3 Validación del Entorno SUMO

**Archivo:** `Distribuciones/validacion_entorno.ipynb`

Ejecuta el simulador SUMO con los archivos generados y verifica:

- Ausencia de errores de conexión en la red.
- Total de vehículos procesados: 2334.
- Distribución por tipo: car (1608), moto (195), bus (531).
- Distribución temporal: pico en intervalo 20-30 minutos.
- Métricas de viaje: tiempo de espera promedio 57.63 s, cola promedio 40.06 vehículos.

---

## 5. Simulación de Línea Base (Tiempos Fijos)

**Archivo:** `Simulaciones_tiempos_fijos/baseline.ipynb`

Implementa el controlador semafórico convencional con ciclos fijos.

### 5.1 Configuración

- **Ciclo:** 120 segundos (55 s verde + 5 s amarillo por fase).
- **Fases:** Fase 0 (Norte-Sur), Fase 1 (Este-Oeste).
- **API:** TraCI / libsumo.
- **Archivo de red:** `Simple_Intersection.net.xml`.
- **Archivo de rutas:** `loja_intersection_weibull.rou.xml`.

### 5.2 Resultados

| Métrica | Valor |
|---|---|
| Vehículos procesados | 2334 |
| Velocidad promedio | 18.17 km/h |
| Tiempo de espera promedio | 57.63 s |
| Tiempo de espera máximo | 117.00 s |
| Tiempo perdido promedio | 89.80 s |
| Cola promedio | 40.06 vehículos |
| Cola máxima | 72 vehículos |

### 5.3 Calibración de Bordes de Discretización

El notebook `baseline.ipynb` extrae los percentiles (33% y 66%) de las series temporales de cola y densidad vehicular. Estos bordes se guardan en `resultados/bin_edges_calibrados.npz` y se utilizan en Q-Learning para discretizar el espacio de estados.

---

## 6. Q-Learning Tabular

**Archivo:** `Q-Learning/qlearning.ipynb`

Implementa un agente Q-Learning con discretización calibrada de estados.

### 6.1 Espacio de Estados

El wrapper `TrafficStateWrapper` (hereda de `gym.Wrapper`) discretiza el vector de observación continuo de sumo-rl usando los bordes calibrados del baseline.

**Componentes del estado (tupla hashable):**

- Densidad por carril: 6 niveles (percentiles 0-33%, 33-66%, 66-100%).
- Cola por carril: 6 niveles.
- Fase actual: 4 niveles.
- Duración acumulada de la fase: 3 niveles.

**Total de estados:** 6 × 6 × 4 × 3 = 288 estados.

### 6.2 Espacio de Acciones

El wrapper `VariableGreenWrapper` define 12 acciones correspondientes a duraciones de verde:

$$A = \{5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60\} \text{ segundos}$$

Cada acción ejecuta internamente múltiples pasos de simulación (delta_time = 5 s) y acumula la recompensa.

### 6.3 Función de Recompensa

$$R_t = -2.0 \cdot W_{prom} + 1.5 \cdot T + 3.0 \cdot I_{espera} + C$$

Donde:

- $W_{prom}$: Tiempo de espera promedio por vehículo.
- $T$: Throughput (vehículos que abandonaron la intersección).
- $I_{espera}$: Mejora en tiempo de espera vs paso anterior.
- $C$: Penalización por congestión (-100 si waiting > 9s AND queue > 61).

### 6.4 Hiperparámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| alpha (tasa de aprendizaje) | 0.1 | Controla la actualización de Q-valores |
| gamma (factor de descuento) | 0.95 | Peso de recompensas futuras |
| Episodios | 5000 | Iteraciones completas de simulación |
| epsilon inicial | 0.5 | Probabilidad de acción aleatoria |
| epsilon mínimo | 0.05 | Exploración residual |
| Decay de epsilon | 0.9995 | Reducción por episodio |
| Replay buffer | 2000 | Almacena experiencias históricas |
| Batch size | 64 | Muestras por actualización |
| Update frequency | 4 | Pasos entre actualizaciones |

### 6.5 Resultados

| Métrica | Valor | Mejora vs Baseline |
|---|---|---|
| Tiempo de espera promedio | 36.89 s | -35.9% |
| Tiempo de espera máximo | 71.00 s | -39.3% |
| Cola promedio | 23.14 veh | -42.2% |
| Cola máxima | 38 veh | -47.2% |

---

## 7. Deep Q-Network (DQN)

**Archivo:** `DQN/dqn.ipynb`

Implementa un agente de aprendizaje profundo utilizando la biblioteca stable-baselines3.

### 7.1 Espacio de Estados

A diferencia de Q-Learning, DQN procesa el vector de estados continuo directamente sin discretización. El vector incluye densidad normalizada, velocidad promedio y presencia de colas en cada carril.

### 7.2 Arquitectura de la Red Neuronal

- **Tipo:** Multi-Layer Perceptron (MLP).
- **Capas ocultas:** [128, 128] neuronas.
- **Activación:** ReLU.
- **Capa de salida:** 12 nodos (una por cada acción de duración de verde).

### 7.3 Hiperparámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| Learning rate | 1.0e-4 | Tasa de aprendizaje del optimizador Adam |
| Buffer size | 100,000 | Capacidad máxima del replay buffer |
| Learning starts | 2000 | Pasos de exploración antes de entrenar |
| Batch size | 128 | Muestras por minilote |
| Target update | 1800 | Frecuencia de sincronización de red objetivo |
| Exploration fraction | 0.8 | Fracción del entrenamiento con decaimiento de epsilon |
| Exploration final eps | 0.05 | Epsilon final |
| Gamma | 0.99 | Factor de descuento |
| Total timesteps | 72,000 | Pasos totales de entrenamiento |
| Arquitectura | [128, 128] | Neuronas por capa oculta |

### 7.4 Función de Recompensa

Misma función multi-objetivo que Q-Learning (sección 6.3).

### 7.5 Resultados

| Métrica | Valor | Mejora vs Baseline |
|---|---|---|
| Tiempo de espera promedio | 17.09 s | -70.3% |
| Tiempo de espera máximo | 38.00 s | -67.5% |
| Cola promedio | 10.44 veh | -73.9% |
| Cola máxima | 21 veh | -70.8% |

---

## 8. Clases Principales

### 8.1 VariableGreenWrapper

**Archivos:** `Q-Learning/qlearning.ipynb`, `DQN/dqn.ipynb`

Wrapper que permite al agente seleccionar la duración del ciclo de verde (5-60 segundos). Internamente:

1. Recibe la acción del agente (índice 0-11).
2. Convierte a duración: `duracion = DURACIONES[action_idx]`.
3. Ejecuta `n_pasos = duracion // delta_time` pasos internos con la fase actual.
4. Acumula la recompensa de todos los pasos.
5. Alterna la fase activa.

### 8.2 Reward (Función de Recompensa)

**Archivos:** `Q-Learning/qlearning.ipynb`, `DQN/dqn.ipynb`

Clase con estado interno que rastrea:

- Tiempo de espera promedio del paso anterior.
- Conjunto de IDs de vehículos del paso anterior (para calcular throughput).

Métodos:

- `reset()`: Reinicia el estado al inicio de cada episodio.
- `__call__(ts)`: Calcula la recompensa dado el TrafficState de SUMO.

### 8.3 EarlyStopWrapper

**Archivo:** `DQN/dqn.ipynb`

Wrapper que trunca el episodio cuando `getMinExpectedNumber() == 0` (no hay vehículos pendientes). Reinicia el RewardTracker al inicio de cada episodio.

### 8.4 TrafficStateWrapper

**Archivo:** `Q-Learning/qlearning.ipynb`

Wrapper que discretiza el espacio de estados continuo usando los bordes calibrados del baseline. Convierte las observaciones a tuplas hashables para la tabla Q.

### 8.5 QLAgentConVisitas

**Archivo:** `Q-Learning/qlearning.ipynb`

Extensión del agente Q-Learning base que incluye:

- Tracking de visitas por estado-acción.
- Early stopping por cobertura de la tabla Q.

---

## 9. Dashboard

**Directorio:** `dashboard/`

Aplicación web basada en Streamlit para visualizar y comparar resultados.

### 9.1 Archivos

| Archivo | Líneas | Descripción |
|---|---|---|
| app.py | 525 | Aplicación principal |
| utils.py | 190 | Funciones de carga y procesamiento |
| run.sh | 11 | Script de inicio |
| requirements.txt | 4 | Dependencias |

### 9.2 Funcionalidades

1. **Selector de escenario:** Tiempos Fijos, Q-Learning, DQN.
2. **Carga de archivos personalizados:** Subida de CSV y XML para reemplazar datos originales.
3. **KPI Cards:** Métricas clave con indicadores de mejora porcentual.
4. **Tabla comparativa:** Los 3 escenarios con formato a 2 decimales y resaltado de mejores valores.
5. **Gráficos interactivos (Plotly):**
   - Evolución temporal de colas y tiempos de espera.
   - Distribución por tipo de vehículo (pie chart + bar chart).
   - Patrón de demanda Weibull (barras + línea de espera promedio).

### 9.3 Ejecución

```bash
cd /home/santiago/Documentos/Tesis/dashboard
./run.sh
```

El dashboard estará disponible en http://localhost:8501.

### 9.4 Persistencia de Archivos

Los archivos subidos se almacenan en `st.session_state` y persisten al cambiar de escenario. El botón "Limpiar archivos personalizados" elimina los archivos cargados y restaura los originales.

---

## 10. Archivos de Salida por Escenario

### 10.1 Tiempos Fijos (`Simulaciones_tiempos_fijos/resultados/`)

| Archivo | Formato | Contenido |
|---|---|---|
| metricas.csv | CSV | Series temporales: Paso_Tiempo, Vehiculos_Activos, Cola_Actual, Velocidad_Promedio, TiempoEspera_Promedio, TiempoPerdido_Promedio, Arrival_Rate |
| tripinfo.xml | XML | Datos por vehículo: id, depart, duration, waitingTime, timeLoss, vType, routeLength |
| bin_edges_calibrados.npz | NumPy | Bordes de percentiles para discretización |

### 10.2 Q-Learning (`Q-Learning/resultados/`)

| Archivo | Formato | Contenido |
|---|---|---|
| qlearning.pkl | Pickle | Tabla Q serializada (dict de numpy arrays) |
| metricas.csv | CSV | step, system_total_stopped, system_mean_waiting_time, system_mean_speed |
| tripinfo.xml | XML | Mismo formato que baseline |

### 10.3 DQN (`DQN/resultados/`)

| Archivo | Formato | Contenido |
|---|---|---|
| best_model.zip | ZIP | Modelo DQN con mejor rendimiento |
| dqn_ultimo.zip | ZIP | Último modelo DQN entrenado |
| metricas.csv | CSV | Mismo formato que Q-Learning |
| tripinfo.xml | XML | Mismo formato que baseline |

---

## 11. Formato de Datos

### 11.1 CSV de Métricas (Tiempos Fijos)

| Paso_Tiempo | Vehiculos_Activos | Cola_Actual | Velocidad_Min | Velocidad_Max | Velocidad_Promedio | TiempoEspera_Promedio | TiempoPerdido_Promedio | Arrival_Rate |
|---|---|---|---|---|---|---|---|---|
| 113.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

### 11.2 CSV de Métricas (Q-Learning / DQN)

| step | system_total_stopped | system_total_waiting_time | system_mean_waiting_time | system_mean_speed | J11_stopped | J11_accumulated_waiting_time | J11_average_speed | agents_total_stopped | agents_total_accumulated_waiting_time |
|---|---|---|---|---|---|---|---|---|---|
| 15.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

### 11.3 XML Tripinfo

```xml
<tripinfo id="veh_0" depart="112.00" departLane="-E4_0" departPos="50.00"
           departSpeed="0.00" departDelay="0.00" arrival="185.00"
           arrivalLane="E3_0" arrivalPos="200.00" arrivalSpeed="13.89"
           duration="73.00" routeLength="607.30" waitingTime="5.00"
           waitingCount="1" stopTime="0.00" timeLoss="15.89"
           vType="car"/>
```

---

## 12. Reproducibilidad

### 12.1 Generación de Demanda

```bash
cd /home/santiago/Documentos/Tesis/Distribuciones
/home/santiago/Documentos/Tesis/.venv/bin/python3 -c "
import json
with open('distribucion_weibull.ipynb') as f:
    nb = json.load(f)
exec(''.join(nb['cells'][2]['source']))  # Celda de generacion
"
```

### 12.2 Ejecución de Simulaciones

Cada notebook contiene las celdas necesarias para ejecutar la simulación completa. Los archivos de resultados se generan automáticamente en las carpetas `resultados/`.

### 12.3 Dashboard

```bash
cd /home/santiago/Documentos/Tesis/dashboard
./run.sh
```

---

## 13. Referencias

[1] P. Wang y W. Ni, "An Enhanced Dueling Double Deep Q-Network With Convolutional Block Attention Module for Traffic Signal Optimization in Deep Reinforcement Learning," *IEEE Access*, vol. 12, 2024.

[2] E. Sarango, "Determinación del Flujo Vehicular en Una Zona Crítica de la Ciudad de Loja Aplicando YOLOv5," Trabajo de Titulación, Universidad Nacional de Loja, 2025.

[3] S. Li, Y. Jiang, y X. Xu, "A Deep Adaptive Traffic Signal Controller With Long-Term Planning Horizon and Spatial-Temporal State Definition Under Dynamic Traffic Fluctuations," *IEEE Access*, vol. 8, 2020.

[4] R. S. Sutton y A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[5] A. Hill et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal of Machine Learning Research*, vol. 22, 2021.

[6] P. A. Lopez et al., "Microscopic Traffic Simulation using SUMO," in *Proceedings of the 21st International Conference on Intelligent Transportation Systems (ITSC)*, IEEE, 2018.

[7] L. N. Alegre, "SUMO-RL: An open-source library for Reinforcement Learning in SUMO," GitHub, 2019.

[8] S. Studer et al., "CRISP-ML(Q): A Industrial Standard Process for Machine Learning with Quality Assurance," *IEEE Software*, vol. 38, 2021.

[9] M. Sewid, E. Saber, y A. S. Al-Bayati, "A Novel Deep Reinforcement Learning Approach to Traffic Signal Control for Multiple Intersections," *Applied Sciences*, vol. 13, 2023.

[10] I.-M. Vlasceanu et al., "Comparative Evaluation of Fuzzy Logic and Q-Learning Algorithms for Adaptive Urban Traffic Signal Control," *Electronics*, vol. 14, 2025.
