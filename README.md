**1. Descripcion del Proyecto**  
Este proyecto compara tres estrategias de control semaforico aplicadas a una interseccion de la ciudad de Loja (Av. Isidro Ayora x 8 de Diciembre):  
1. **Tiempos Fijos (Baseline):** Controlador convencional con ciclos predefinidos de 120 segundos.  
2. **Q-Learning Tabular:** Agente de aprendizaje por refuerzo que discretiza el estado de trafico en 288 estados y 12 acciones.  
3. **Deep Q-Network (DQN):** Agente de aprendizaje profundo que procesa el estado continuo mediante una red neuronal MLP [128, 128].  
Los tres escenarios utilizan la misma demanda vehicular generada estocasticamente (distribucion de Weibull, 2418 vehiculos en 3600 segundos) y la misma funcion de recompensa multi-objetivo.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCkLfE07YGfHAiAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse4eQF6VhvmPsAAAAASUVORK5CYII=)  
**2. Estructura del Directorio**  
Tesis/  
 ├── Simple_Intersection.net.xml          # Red de SUMO (netedit 1.27.0)  
 ├── Distribuciones/  
 │   ├── distribucion_weibull.ipynb       # Generacion de demanda estocastica  
 │   ├── validacion_entorno.ipynb         # Validacion de archivos SUMO  
 │   └── loja_intersection_weibull.rou.xml # Archivo de rutas generado  
 ├── Simulaciones_tiempos_fijos/  
 │   ├── baseline.ipynb                   # Simulacion de linea base  
 │   └── resultados/  
 │       ├── metricas.csv                 # Metricas temporales  
 │       ├── tripinfo.xml                 # Info de viajes por vehiculo  
 │       └── bin_edges_calibrados.npz     # Limites de discretizacion  
 ├── Q-Learning/  
 │   ├── qlearning.ipynb                  # Entrenamiento Q-Learning  
 │   └── resultados/  
 │       ├── qlearning.pkl                # Tabla Q serializada  
 │       ├── metricas.csv  
 │       └── tripinfo.xml  
 ├── DQN/  
 │   ├── dqn.ipynb                        # Entrenamiento DQN  
 │   └── resultados/  
 │       ├── best_model.zip               # Mejor modelo DQN  
 │       ├── dqn_ultimo.zip               # Ultimo modelo DQN  
 │       ├── metricas.csv  
 │       └── tripinfo.xml  
 ├── Tratamiento_Datos/  
 │   ├── procesamiento_aforos.ipynb       # Procesamiento de conteos vehiculares  
 │   └── BD_2024_4-1/                     # Dataset de deteccion de objetos  
 ├── dashboard/  
 │   ├── app.py                           # Aplicacion Streamlit  
 │   ├── utils.py                         # Funciones de carga de datos  
 │   ├── run.sh                           # Script de inicio  
 │   ├── requirements.txt  
 │   └── README.md  
 ├── .venv/                               # Entorno virtual Python 3.14.5  
 └── .gitignore  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvkwZyCKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6feBdUJcyS2AAAAAElFTkSuQmCC)  
**3. Requisitos Previos**  
**3.1 Software**  
- **Python 3.14.5** (o superior)  
- **SUMO 1.26.0** o superior (con SUMO_HOME configurado)  
- **Git** (para clonar el repositorio)  
**3.2 Dependencias de Python**  
Las dependencias estan gestionadas en /home/santiago/Documentos/Tesis/.venv/. Paquetes principales:  
| | | |  
|-|-|-|  
| **Paquete** | **Version** | **Uso** |   
| sumo-rl | 1.2.x | Entorno de simulacion para RL |   
| stable-baselines3 | 2.x | Implementacion de DQN |   
| gymnasium | 0.29.x | Framework de entornos RL |   
| pandas | 2.1.x | Manipulacion de datos |   
| numpy | 1.26.x | Calculo numerico |   
| torch | 2.x | Redes neuronales (DQN) |   
| streamlit | 1.59.x | Dashboard web |   
| plotly | 5.x | Graficos interactivos |   
| lxml | 5.x | Parsing de XML |   
   
Para instalar las dependencias del dashboard:  
/home/santiago/Documentos/Tesis/.venv/bin/pip install streamlit plotly pandas lxml  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhQAQ60PcrIhnxgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseS14EKxPCORkAAAAASUVORK5CYII=)  
**4. Pipeline de Datos**  
**4.1 Procesamiento de Aforos Vehiculares**  
**Archivo:** Tratamiento_Datos/procesamiento_aforos.ipynb  
Este notebook procesa los conteos empiricos de vehiculos realizados a partir de grabaciones de video en la interseccion (datos de la tesis de Sarango [2025]). Los pasos son:  
1. Lectura de registros de conteo vehicular por tipo (car, moto, bus).  
2. Calculo de tasas de flujo (veh/s, veh/min, veh/h).  
3. Clasificacion por acceso (Norte via -E5, Oeste via -E4).  
4. Descarga del dataset de imagenes etiquetadas desde Roboflow (3 clases: car, moto, bus).  
**Resultado:** Distribucion por tipo de vehiculo: 74.1% car, 14.8% moto, 11.1% bus. Demanda de diseno: 1205 vehiculos/hora por acceso.  
**4.2 Generacion de Demanda Estocastica**  
**Archivo:** Distribuciones/distribucion_weibull.ipynb  
Genera el archivo de rutas loja_intersection_weibull.rou.xml utilizando una distribucion de Weibull para modelar la variabilidad temporal de las llegadas.  
**Funcion de densidad de probabilidad:**  
\lambda(t) = \frac{k}{\lambda_{w}} \left( \frac{t}{\lambda_{w}} \right)^{k-1} e^{-\left( \frac{t}{\lambda_{w}} \right)^k}  
**Parametros:**  
- k = 2.5 (forma): Controla la asimetrica de la curva.  
- \lambda_{w} = 0.55 \times 3600 (escala): Localiza el pico entre 20-30 minutos.  
- Total de vehiculos: 2418.  
- Duracion: 3600 segundos.  
- Seed: 42 (reproducibilidad).  
**Rutas generadas (4 movimientos):**  
- Norte -> Sur (directo)  
- Norte -> Este (giro izquierdo)  
- Oeste -> Este (directo)  
- Oeste -> Sur (giro derecho)  
**4.3 Validacion del Entorno SUMO**  
**Archivo:** Distribuciones/validacion_entorno.ipynb  
Ejecuta el simulador SUMO con los archivos generados y verifica:  
- Ausencia de errores de conexion en la red.  
- Total de vehiculos procesados: 2418.  
- Distribucion por tipo: car (1791), moto (358), bus (269).  
- Distribucion temporal: pico en intervalo 20-30 minutos.  
- Metricas de viaje: tiempo de espera promedio 57.63s, cola promedio 40.06 vehiculos.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpg/h5VMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA224BcUMk6pDAAAAAElFTkSuQmCC)  
**5. Simulacion de Linea Base (Tiempos Fijos)**  
**Archivo:** Simulaciones_tiempos_fijos/baseline.ipynb  
Implementa el controlador semaforico convencional con ciclos fijos.  
**5.1 Configuracion**  
- **Ciclo:** 120 segundos (55s verde + 5s amarillo por fase).  
- **Fases:** Fase 0 (Norte-Sur), Fase 1 (Este-Oeste).  
- **API:** TraCI / libsumo.  
- **Archivos de red:**Simple_Intersection.net.xml.  
- **Archivo de rutas:**loja_intersection_weibull.rou.xml.  
**5.2 Resultados**  
| | |  
|-|-|  
| **Metrica** | **Valor** |   
| Vehiculos procesados | 2418 |   
| Velocidad promedio | 18.17 km/h |   
| Tiempo de espera promedio | 57.63 s |   
| Tiempo de espera maximo | 117.00 s |   
| Tiempo perdido promedio | 89.80 s |   
| Cola promedio | 40.06 vehiculos |   
| Cola maxima | 72 vehiculos |   
   
**5.3 Calibracion de Bordes de Discretizacion**  
El notebook baseline.ipynb extrae los percentiles (33% y 66%) de las series temporales de cola y densidad vehicular. Estos bordes se guardan en resultados/bin_edges_calibrados.npz y se utilizan en Q-Learning para discretizar el espacio de estados.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAUBBAwSfIb+HdmNvAkgaxgjcRZhLMNjNHdQUAwF/ce7Wq8+sJAACvrQctewNKtdojwQAAAABJRU5ErkJggg==)  
**6. Q-Learning Tabular**  
**Archivo:** Q-Learning/qlearning.ipynb  
Implementa un agente Q-Learning con discretizacion calibrada de estados.  
**6.1 Espacio de Estados**  
El wrapper TrafficStateWrapper (hereda de gym.Wrapper) discretiza el vector de observacion continuo de sumo-rl usando los bordes calibrados del baseline.  
**Componentes del estado (tupla hashable):**  
- Densidad por carril: 6 niveles (percentiles 0-33%, 33-66%, 66-100%).  
- Cola por carril: 6 niveles.  
- Fase actual: 4 niveles.  
- Duracion acumulada de la fase: 3 niveles.  
**Total de estados:** 6 x 6 x 4 x 3 = 288 estados.  
**6.2 Espacio de Acciones**  
El wrapper VariableGreenWrapper define 12 acciones correspondientes a duraciones de verde:  
A = \{5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60\} \text{ segundos}  
Cada accion ejecuta internamente multiples pasos de simulacion (delta_time = 5s) y acumula la recompensa.  
**6.3 Funcion de Recompensa**  
R_t = -2.0 \cdot W_{prom} + 1.5 \cdot T + 3.0 \cdot I_{espera} + C  
Donde:  
- W_{prom}: Tiempo de espera promedio por vehiculo.  
- T: Throughput (vehiculos que abandonaron la interseccion).  
- I_{espera}: Mejora en tiempo de espera vs paso anterior.  
- C: Penalizacion por congestion (-100 si waiting > 9s AND queue > 61).  
**6.4 Hiperparametros**  
| | | |  
|-|-|-|  
| **Parametro** | **Valor** | **Descripcion** |   
| alpha (tasa de aprendizaje) | 0.1 | Controla la actualizacion de Q-valores |   
| gamma (factor de descuento) | 0.95 | Peso de recompensas futuras |   
| Episodios | 5000 | Iteraciones completas de simulacion |   
| epsilon inicial | 0.5 | Probabilidad de accion aleatoria |   
| epsilon minimo | 0.05 | Exploracion residual |   
| Decay de epsilon | 0.9995 | Reduccion por episodio |   
| Replay buffer | 2000 | Almacena experiencias historicas |   
| Batch size | 64 | Muestras por actualizacion |   
| Update frequency | 4 | Pasos entre actualizaciones |   
   
**6.5 Resultados**  
| | | |  
|-|-|-|  
| **Metrica** | **Valor** | **Mejora vs Baseline** |   
| Tiempo de espera promedio | 36.89 s | -35.9% |   
| Tiempo de espera maximo | 71.00 s | -39.3% |   
| Cola promedio | 23.14 veh | -42.2% |   
| Cola maxima | 38 veh | -47.2% |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpg/h5VMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA224BcUMk6pDAAAAAElFTkSuQmCC)  
**7. Deep Q-Network (DQN)**  
**Archivo:** DQN/dqn.ipynb  
Implementa un agente de aprendizaje profundo utilizando la biblioteca stable-baselines3.  
**7.1 Espacio de Estados**  
A diferencia de Q-Learning, DQN procesa el vector de estados continuo directamente sin discretizacion. El vector incluye densidad normalizada, velocidad promedio y presencia de colas en cada carril.  
**7.2 Arquitectura de la Red Neuronal**  
- **Tipo:** Multi-Layer Perceptron (MLP).  
- **Capas ocultas:** [128, 128] neuronas.  
- **Activacion:** ReLU.  
- **Capa de salida:** 12 nodos (una por cada accion de duracion de verde).  
**7.3 Hiperparametros**  
| | | |  
|-|-|-|  
| **Parametro** | **Valor** | **Descripcion** |   
| Learning rate | 1.0e-4 | Tasa de aprendizaje del optimizador Adam |   
| Buffer size | 100,000 | Capacidad maxima del replay buffer |   
| Learning starts | 2000 | Pasos de exploracion antes de entrenar |   
| Batch size | 128 | Muestras por minilote |   
| Target update | 1800 | Frecuencia de sincronizacion de red objetivo |   
| Exploration fraction | 0.8 | Fraccion del entrenamiento con decaimiento de epsilon |   
| Exploration final eps | 0.05 | Epsilon final |   
| Gamma | 0.99 | Factor de descuento |   
| Total timesteps | 72,000 | Pasos totales de entrenamiento |   
| Arquitectura | [128, 128] | Neuronas por capa oculta |   
   
**7.4 Funcion de Recompensa**  
Misma funcion multi-objetivo que Q-Learning (seccion 6.3).  
**7.5 Resultados**  
| | | |  
|-|-|-|  
| **Metrica** | **Valor** | **Mejora vs Baseline** |   
| Tiempo de espera promedio | 17.09 s | -70.3% |   
| Tiempo de espera maximo | 38.00 s | -67.5% |   
| Cola promedio | 10.44 veh | -73.9% |   
| Cola maxima | 21 veh | -70.8% |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBACPq8MH2NpGACyywEZJWQZeZ2aszAAD+4l6rrTq+ngAA8Nr1AL/KBEe6dElaAAAAAElFTkSuQmCC)  
**8. Clases Principales**  
**8.1 VariableGreenWrapper**  
**Archivos:** Q-Learning/qlearning.ipynb, DQN/dqn.ipynb  
Wrapper que permite al agente seleccionar la duracion del ciclo de verde (5-60 segundos). Internamente:  
1. Recibe la accion del agente (indice 0-11).  
2. Convierte a duracion: duracion = DURACIONES[action_idx].  
3. Ejecuta n_pasos = duracion // delta_time pasos internos con la fase actual.  
4. Acumula la recompensa de todos los pasos.  
5. Alterna la fase activa.  
**8.2 Reward (Funcion de Recompensa)**  
**Archivos:** Q-Learning/qlearning.ipynb, DQN/dqn.ipynb  
Clase con estado interno que rastrea:  
- Tiempo de espera promedio del paso anterior.  
- Conjunto de IDs de vehiculos del paso anterior (para calcular throughput).  
Metodos:  
- reset(): Reinicia el estado al inicio de cada episodio.  
- __call__(ts): Calcula la recompensa dado el TrafficState de SUMO.  
**8.3 EarlyStopWrapper**  
**Archivo:** DQN/dqn.ipynb  
Wrapper que trunca el episodio cuando getMinExpectedNumber() == 0 (no hay vehiculos pendientes). Reinicia el RewardTracker al inicio de cada episodio.  
**8.4 TrafficStateWrapper**  
**Archivo:** Q-Learning/qlearning.ipynb  
Wrapper que discretiza el espacio de estados continuo usando los bordes calibrados del baseline. Convierte las observaciones a tuplas hashables para la tabla Q.  
**8.5 QLAgentConVisitas**  
**Archivo:** Q-Learning/qlearning.ipynb  
Extension del agente Q-Learning base que incluye:  
- Tracking de visitas por estado-accion.  
- Early stopping por cobertura de la tabla Q.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/jkUsYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4rDBc72meO5AAAAAElFTkSuQmCC)  
**9. Dashboard**  
**Directorio:** dashboard/  
Aplicacion web basada en Streamlit para visualizar y comparar resultados.  
**9.1 Archivos**  
| | | |  
|-|-|-|  
| **Archivo** | **Lineas** | **Descripcion** |   
| app.py | 525 | Aplicacion principal |   
| utils.py | 190 | Funciones de carga y procesamiento |   
| run.sh | 11 | Script de inicio |   
| requirements.txt | 4 | Dependencias |   
   
**9.2 Funcionalidades**  
1. **Selector de escenario:** Tiempos Fijos, Q-Learning, DQN.  
2. **Carga de archivos personalizados:** Subida de CSV y XML para reemplazar datos originales.  
3. **KPI Cards:** Metricas clave con indicadores de mejora porcentual.  
4. **Tabla comparativa:** Los 3 escenarios con formato a 2 decimales y resaltado de mejores valores.  
5. **Graficos interactivos (Plotly):**  
  - Evolucion temporal de colas y tiempos de espera.  
  - Distribucion por tipo de vehiculo (pie chart + bar chart).  
  - Patron de demanda Weibull (barras + linea de espera promedio).  
**9.3 Ejecucion**  
cd /home/santiago/Documentos/Tesis/dashboard  
 ./run.sh  
   
El dashboard estara disponible en http://localhost:8501.  
**9.4 Persistencia de Archivos**  
Los archivos subidos se almacenan en st.session_state y persisten al cambiar de escenario. El boton "Limpiar archivos personalizados" elimina los archivos cargados y restaura los originales.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZscZXlheJwqQgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseop8EQrmJduIAAAAASUVORK5CYII=)  
**10. Archivos de Salida por Escenario**  
**10.1 Tiempos Fijos (**Simulaciones_tiempos_fijos/resultados/ **)**  
| | | |  
|-|-|-|  
| **Archivo** | **Formato** | **Contenido** |   
| metricas.csv | CSV | Series temporales: Paso_Tiempo, Vehiculos_Activos, Cola_Actual, Velocidad_Promedio, TiempoEspera_Promedio, TiempoPerdido_Promedio, Arrival_Rate |   
| tripinfo.xml | XML | Datos por vehiculo: id, depart, duration, waitingTime, timeLoss, vType, routeLength |   
| bin_edges_calibrados.npz | NumPy | Bordes de percentiles para discretizacion |   
   
**10.2 Q-Learning (**Q-Learning/resultados/ **)**  
| | | |  
|-|-|-|  
| **Archivo** | **Formato** | **Contenido** |   
| qlearning.pkl | Pickle | Tabla Q serializada (dict de numpy arrays) |   
| metricas.csv | CSV | step, system_total_stopped, system_mean_waiting_time, system_mean_speed |   
| tripinfo.xml | XML | Mismo formato que baseline |   
   
**10.3 DQN (**DQN/resultados/ **)**  
| | | |  
|-|-|-|  
| **Archivo** | **Formato** | **Contenido** |   
| best_model.zip | ZIP | Modelo DQN con mejor rendimiento |   
| dqn_ultimo.zip | ZIP | Ultimo modelo DQN entrenado |   
| metricas.csv | CSV | Mismo formato que Q-Learning |   
| tripinfo.xml | XML | Mismo formato que baseline |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZKQ6kBqjSAOJywYYCIkd9OP36pqRMQMAAB+sfqJfLoBAMCN3NYoAzBA+QG0AAAAAElFTkSuQmCC)  
**11. Formato de Datos**  
**11.1 CSV de Metricas (Tiempos Fijos)**  
Paso_Tiempo,Vehiculos_Activos,Cola_Actual,Velocidad_Min,Velocidad_Max,Velocidad_Promedio,TiempoEspera_Promedio,TiempoPerdido_Promedio,Arrival_Rate  
 113.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0  
   
**11.2 CSV de Metricas (Q-Learning / DQN)**  
step,system_total_stopped,system_total_waiting_time,system_mean_waiting_time,system_mean_speed,J11_stopped,J11_accumulated_waiting_time,J11_average_speed,agents_total_stopped,agents_total_accumulated_waiting_time  
 15.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0  
   
**11.3 XML Tripinfo**  
<tripinfo id="veh_0" depart="112.00" departLane="-E4_0" departPos="50.00"  
           departSpeed="0.00" departDelay="0.00" arrival="185.00"  
           arrivalLane="E3_0" arrivalPos="200.00" arrivalSpeed="13.89"  
           duration="73.00" routeLength="607.30" waitingTime="5.00"  
           waitingCount="1" stopTime="0.00" timeLoss="15.89"  
           vType="car"/>  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OUQmAABBAsSeYxZyXSzCJASxgACv4J8KWYMvMbNURAAB/ca7VXe1fTwAAeO16AKe+BdmJqrPdAAAAAElFTkSuQmCC)  
**12. Reproducibilidad**  
**12.1 Generacion de Demand**  
cd /home/santiago/Documentos/Tesis/Distribuciones  
 /home/santiago/Documentos/Tesis/.venv/bin/python3 -c "  
 import json  
 with open('distribucion_weibull.ipynb') as f:  
     nb = json.load(f)  
 exec(''.join(nb['cells'][2]['source']))  # Celda de generacion  
 "  
   
**12.2 Ejecucion de Simulaciones**  
Cada notebook contiene las celdas necesarias para ejecutar la simulacion completa. Los archivos de resultados se generan automaticamente en las carpetas resultados/.  
**12.3 Dashboard**  
cd /home/santiago/Documentos/Tesis/dashboard  
 ./run.sh  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OUQmAABBAsSeIWMICprwEpjSIFfwTYUuwZWaO6goAgL+412qrzq8nAAC8tj8tdQNNdXaCdAAAAABJRU5ErkJggg==)  
**13. Referencias**  
[1] P. Wang y W. Ni, "An Enhanced Dueling Double Deep Q-Network With Convolutional Block Attention Module for Traffic Signal Optimization in Deep Reinforcement Learning," IEEE Access, vol. 12, 2024.  
[2] E. Sarango, "Determinacion del Flujo Vehicular en Una Zona Critica de la Ciudad de Loja Aplicando YOLOv5," Trabajo de Titulacion, Universidad Nacional de Loja, 2025.  
[3] S. Li, Y. Jiang, y X. Xu, "A Deep Adaptive Traffic Signal Controller With Long-Term Planning Horizon and Spatial-Temporal State Definition Under Dynamic Traffic Fluctuations," IEEE Access, vol. 8, 2020.  
[4] R. S. Sutton y A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.  
[5] A. Hill et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," Journal of Machine Learning Research, vol. 22, 2021.  
[6] P. A. Lopez et al., "Microscopic Traffic Simulation using SUMO," in Proceedings of the 21st International Conference on Intelligent Transportation Systems (ITSC), IEEE, 2018.  
[7] L. N. Alegre, "SUMO-RL: An open-source library for Reinforcement Learning in SUMO," GitHub, 2019.  
[8] S. Studer et al., "CRISP-ML(Q): A Industrial Standard Process for Machine Learning with Quality Assurance," IEEE Software, vol. 38, 2021.  
[9] M. Sewid, E. Saber, y A. S. Al-Bayati, "A Novel Deep Reinforcement Learning Approach to Traffic Signal Control for Multiple Intersections," Applied Sciences, vol. 13, 2023.  
[10] I.-M. Vlasceanu et al., "Comparative Evaluation of Fuzzy Logic and Q-Learning Algorithms for Adaptive Urban Traffic Signal Control," Electronics, vol. 14, 2025.  
