# Plan: Reorganizar Estructura de Experimentos

## Estructura final

```
Simulaciones_Q-Learning/
  ql_puro/
    qlearning_exp1_no_discret.ipynb
    resultados/
      modelo_qlearning.pkl
      tripinfo_final.xml
      dataset_sumo.csv
  ql_discr/
    qlearning_exp2_discr.ipynb
    resultados/
      modelo_qlearning.pkl
      tripinfo_final.xml
      dataset_sumo.csv
  ql_doble/
    qlearning_exp3_rec_doble.ipynb
    resultados/
      modelo_qlearning.pkl
      tripinfo_final.xml
      dataset_sumo.csv

Simulaciones_DQN/
  dqn_doble/
    dqn_weibull.ipynb
    resultados/
      best_model.zip
      last_model.zip
      tripinfo_final.xml
      dataset_sumo.csv
```

## Pasos de implementacion

### Paso 1: Crear carpetas nuevas

```
Simulaciones_Q-Learning/ql_puro/resultados/
Simulaciones_Q-Learning/ql_discr/resultados/
Simulaciones_Q-Learning/ql_doble/resultados/
Simulaciones_DQN/dqn_doble/resultados/
```

### Paso 2: Mover notebooks

```
Exp1/qlearning_exp1_no_discret.ipynb  -> ql_puro/qlearning_exp1_no_discret.ipynb
Exp2/qlearning_exp2_discr.ipynb       -> ql_discr/qlearning_exp2_discr.ipynb
Exp3/qlearning_exp3_rec_doble.ipynb   -> ql_doble/qlearning_exp3_rec_doble.ipynb
Simulaciones_DQN/dqn_weibull.ipynb    -> dqn_doble/dqn_weibull.ipynb
```

### Paso 3: Mover modelos existentes a resultados/

```
Exp1/ejecuciones/modelo_qlearning.pkl -> ql_puro/resultados/modelo_qlearning.pkl
Exp2/ejecuciones/modelo_qlearning.pkl -> ql_discr/resultados/modelo_qlearning.pkl
Exp3/ejecuciones/modelo_qlearning.pkl -> ql_doble/resultados/modelo_qlearning.pkl
```

### Paso 4: Mover tripinfo de validacion existentes

```
Exp2/ejecuciones/exp2_tripinfo_final.xml -> ql_discr/resultados/tripinfo_final.xml
Exp3/ejecuciones/exp3_tripinfo_final.xml -> ql_doble/resultados/tripinfo_final.xml
```

### Paso 5: Eliminar carpetas viejas

```
Exp1/ (vacia despues de mover)
Exp2/ejecuciones/ (borrar tripinfo XMLs de entrenamiento)
Exp3/ejecuciones/ (borrar tripinfo XMLs de entrenamiento)
Simulaciones_DQN/resultados_dqn/ (si existe)
```

### Paso 6: Actualizar notebooks Q-Learning (Exp1, Exp2, Exp3)

En cada notebook aplicar estos cambios:

**a) Rutas de archivos SUMO (ajustar profundidad):**
```python
# ANTES (dentro de Exp1/):
NET = "..\\Simple_Intersection.net.xml"
ROU = "..\\Distribuciones\\Dist_Weibull_V2\\loja_intersection_weibull.rou.xml"

# DESPUES (dentro de ql_puro/ - misma profundidad, sin cambio):
NET = "..\\Simple_Intersection.net.xml"
ROU = "..\\Distribuciones\\Dist_Weibull_V2\\loja_intersection_weibull.rou.xml"
```

**b) OUT_FOLDER:**
```python
# Exp1:
OUT_FOLDER = "resultados"
# Exp2:
OUT_FOLDER = "resultados"
# Exp3:
OUT_FOLDER = "resultados"
```

**c) Training loop - eliminar exports innecesarios:**
```python
# ELIMINAR estas lineas:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"{OUT_FOLDER}/run{run}_ep{episode}_{timestamp}"
archivo_csv = f"{base_name}_metricas"
archivo_xml = f"{base_name}_tripinfo.xml"

# CAMBIAR SumoEnvironment:
# ANTES:
env = SumoEnvironment(
    ...
    out_csv_name=archivo_csv,
    ...
    additional_sumo_cmd=f"--tripinfo-output {archivo_xml}"
)
# DESPUES:
env = SumoEnvironment(
    ...
    out_csv_name="temp",
    ...
    # SIN additional_sumo_cmd
)
```

**d) Validacion - actualizar rutas:**
```python
# ANTES (Exp1):
env_eval = SumoEnvironment(
    ...
    out_csv_name=f'{OUT_FOLDER}/evaluacion_final',
    ...
    additional_sumo_cmd=f"--tripinfo-output {OUT_FOLDER}/exp1_tripinfo_final.xml"
)
# ...
df_metricas.to_csv(f"{OUT_FOLDER}/exp1_dataset_sumo.csv", index=False)

# DESPUES (Exp1):
env_eval = SumoEnvironment(
    ...
    out_csv_name=f'{OUT_FOLDER}/evaluacion_final',
    ...
    additional_sumo_cmd=f"--tripinfo-output {OUT_FOLDER}/tripinfo_final.xml"
)
# ...
df_metricas.to_csv(f"{OUT_FOLDER}/dataset_sumo.csv", index=False)
```

**e) Celdas de analisis - actualizar rutas:**
```python
# ANTES:
ruta_xml = f"{OUT_FOLDER}/exp1_tripinfo_final.xml"
df_metricas = pd.read_csv(f"{OUT_FOLDER}/exp1_dataset_sumo.csv")

# DESPUES:
ruta_xml = f"{OUT_FOLDER}/tripinfo_final.xml"
df_metricas = pd.read_csv(f"{OUT_FOLDER}/dataset_sumo.csv")
```

**f) Modelo - actualizar ruta:**
```python
# ANTES:
ruta_modelo = f"{OUT_FOLDER}/modelo_qlearning.pkl"

# DESPUES (sin cambio, ya usa OUT_FOLDER):
ruta_modelo = f"{OUT_FOLDER}/modelo_qlearning.pkl"
```

### Paso 7: Actualizar notebook DQN

**a) OUT_FOLDER:**
```python
# ANTES:
OUT_FOLDER = "resultados_dqn"
# DESPUES:
OUT_FOLDER = "resultados"
```

**b) Rutas de archivos SUMO (ajustar profundidad):**
```python
# ANTES (dentro de Simulaciones_DQN/):
NET = "..\\Simple_Intersection.net.xml"
ROU = "..\\Distribuciones\\Dist_Weibull_V2\\loja_intersection_weibull.rou.xml"

# DESPUES (dentro de dqn_doble/ - misma profundidad, sin cambio):
NET = "..\\Simple_Intersection.net.xml"
ROU = "..\\Distribuciones\\Dist_Weibull_V2\\loja_intersection_weibull.rou.xml"
```

**c) Validacion - actualizar nombres de archivo:**
```python
# ANTES:
additional_sumo_cmd=f"--tripinfo-output {OUT_FOLDER}/dqn_tripinfo_final.xml"
df_metricas.to_csv(f"{OUT_FOLDER}/dqn_dataset_sumo.csv", index=False)

# DESPUES:
additional_sumo_cmd=f"--tripinfo-output {OUT_FOLDER}/tripinfo_final.xml"
df_metricas.to_csv(f"{OUT_FOLDER}/dataset_sumo.csv", index=False)
```

**d) Celdas de analisis - actualizar rutas:**
```python
# ANTES:
ruta_xml = f"{OUT_FOLDER}/dqn_tripinfo_final.xml"
df_metricas = pd.read_csv(f"{OUT_FOLDER}/dqn_dataset_sumo.csv")

# DESPUES:
ruta_xml = f"{OUT_FOLDER}/tripinfo_final.xml"
df_metricas = pd.read_csv(f"{OUT_FOLDER}/dataset_sumo.csv")
```

## Resumen de archivos por experimento

### ql_puro/resultados/
- `modelo_qlearning.pkl` - modelo entrenado
- `tripinfo_final.xml` - datos de vehiculos (validacion)
- `dataset_sumo.csv` - metricas paso a paso (validacion)
- `evaluacion_final_conn1_ep1.csv` - auto-generado por sumo-rl (se puede ignorar)

### ql_discr/resultados/
- Mismos archivos que ql_puro

### ql_doble/resultados/
- Mismos archivos que ql_puro

### dqn_doble/resultados/
- `best_model.zip` - mejor modelo DQN
- `last_model.zip` - ultimo modelo DQN
- `tripinfo_final.xml` - datos de vehiculos (validacion)
- `dataset_sumo.csv` - metricas paso a paso (validacion)

## Verificacion

1. Verificar estructura de carpetas creada correctamente
2. Ejecutar cada notebook desde su nueva ubicacion
3. Confirmar que resultados/ solo contiene archivos de validacion (no de entrenamiento)
4. Confirmar que modelos existentes se cargan correctamente
