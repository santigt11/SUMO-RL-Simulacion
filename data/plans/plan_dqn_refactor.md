# Plan: Refactorizacion DQN Notebook

## Contexto
El notebook DQN actual (`Simulaciones_DQN/dqn_weibull.ipynb`) no exporta tripinfo XML ni CSV de datos paso a paso. Debe replicar la estructura de Q-Learning (`qlearning.ipynb` y `Exp3/qlearning_exp3_rec_doble.ipynb`) pero usando DQN de stable-baselines3 en lugar de Q-Learning tabular.

## Decisiones del usuario
1. **Timesteps**: Mantener 72000 total
2. **num_seconds**: 100000 como max, con parada dinamica cuando red se vacie (`getMinExpectedNumber() == 0`)
3. **Recompensa**: Doble recompensa del Exp3: `-(1.0 * cola + 0.1 * espera_total)`
4. **Modelos**: `best_model` (mejor reward acumulado) + `last_model` (ultimo episodio)
5. **Exports**: Solo de validacion (tripinfo XML + CSV). Sin exports de entrenamiento.

## Reto tecnico
SB3 usa `model.learn()` internamente, no tiene loop manual como Q-Learning. La parada dinamica requiere un wrapper que trunque el episodio cuando la red se vacie.

## Solucion: EarlyStopWrapper
```python
class EarlyStopWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.sumo.simulation.getMinExpectedNumber() == 0:
            truncated = True
        return obs, reward, terminated, truncated, info
```
Esto hace que SB3 termine el episodio cuando la red se vacie, igual que el `break` en Q-Learning.

## Estructura del notebook (10 celdas)

### Celda 1: Markdown
Titulo + descripcion del experimento DQN con doble recompensa.

### Celda 2: Imports
```python
import argparse
from datetime import datetime
import pandas as pd
import os
import sys
import torch
torch._dynamo.config.suppress_errors = True
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
```

### Celda 3: SUMO_HOME + Recompensa doble + Wrapper
```python
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME no definido")

def recompensa_combinada(ts):
    cola = ts.get_total_queued()
    espera_total = sum(ts.get_accumulated_waiting_time_per_lane())
    return -(1.0 * cola + 0.1 * espera_total)

class EarlyStopWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.sumo.simulation.getMinExpectedNumber() == 0:
            truncated = True
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
```

### Celda 4: Paths + get_args()
```python
NET = "..\\Simple_Intersection.net.xml"
ROU = "..\\Distribuciones\\Dist_Weibull_V2\\loja_intersection_weibull.rou.xml"
OUT_FOLDER = "resultados_dqn"
os.makedirs(OUT_FOLDER, exist_ok=True)

def get_args():
    prs = argparse.ArgumentParser(...)
    prs.add_argument("-route", default=ROU)
    prs.add_argument("-net", default=NET)
    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-total_timesteps", type=int, default=72000)
    prs.add_argument("-num_seconds", type=int, default=100000)
    prs.add_argument("-lr", type=float, default=1e-3)
    prs.add_argument("-buffer_size", type=int, default=50000)
    prs.add_argument("-learning_starts", type=int, default=1000)
    prs.add_argument("-target_update", type=int, default=500)
    prs.add_argument("-exploration_fraction", type=float, default=0.5)
    prs.add_argument("-exploration_final_eps", type=float, default=0.05)
    return prs.parse_args(args=[])
```

### Celda 5: Training loop
```python
args = get_args()
historial_recompensas = []
mejor_recompensa = float('-inf')

# Crear primer entorno para inicializar el modelo
env_train = SumoEnvironment(
    net_file=args.net, route_file=args.route,
    out_csv_name="temp", single_agent=True,
    use_gui=args.gui, num_seconds=args.num_seconds,
    reward_fn=recompensa_combinada, time_to_teleport=-1
)
env_train = EarlyStopWrapper(env_train)

model = DQN("MlpPolicy", env_train,
    learning_rate=args.lr, buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    target_update_interval=args.target_update,
    exploration_fraction=args.exploration_fraction,
    exploration_final_eps=args.exploration_final_eps,
    verbose=1)

print("Iniciando entrenamiento DQN...")
episode = 0

while model.num_timesteps < args.total_timesteps:
    episode += 1
    model.learn(total_timesteps=args.num_seconds, reset_num_timesteps=False)
    
    ep_reward = model.ep_info_buffer[-1]['r'] if model.ep_info_buffer else 0
    
    historial_recompensas.append(ep_reward)
    model.save(f"{OUT_FOLDER}/last_model")
    
    if ep_reward > mejor_recompensa:
        mejor_recompensa = ep_reward
        model.save(f"{OUT_FOLDER}/best_model")
    
    print(f"Ep {episode:02d} | Reward: {ep_reward:.2f} | Mejor: {mejor_recompensa:.2f} | Timesteps: {model.num_timesteps}")
    
    env_train.close()
    
    if model.num_timesteps < args.total_timesteps:
        env_train = SumoEnvironment(
            net_file=args.net, route_file=args.route,
            out_csv_name="temp", single_agent=True,
            use_gui=args.gui, num_seconds=args.num_seconds,
            reward_fn=recompensa_combinada, time_to_teleport=-1)
        env_train = EarlyStopWrapper(env_train)
        model.set_env(env_train)

env_train.close()
print(f"Entrenamiento finalizado. Episodios: {episode}")
```

**Nota**: `model.ep_info_buffer` contiene las recompensas de los episodios completados. Si el wrapper trunca con `truncated=True`, SB3 registra la info correctamente.

### Celda 6: Learning curve
```python
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(historial_recompensas) + 1), historial_recompensas, marker='o', color='green')
plt.title("Curva de Aprendizaje: DQN + Doble Recompensa")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Acumulada")
plt.show()
```

### Celda 7: Evaluation (validacion)
Cargar `best_model`, crear entorno con tripinfo + CSV manual. Sin DiscretizedObservationWrapper.
```python
from stable_baselines3 import DQN

print("Cargando mejor modelo...")
model = DQN.load(f"{OUT_FOLDER}/best_model")

env_eval = SumoEnvironment(
    net_file=args.net, route_file=args.route,
    out_csv_name=f'{OUT_FOLDER}/evaluacion_final',
    single_agent=True, use_gui=False,
    num_seconds=args.num_seconds,
    time_to_teleport=-1,
    additional_sumo_cmd=f"--tripinfo-output {OUT_FOLDER}/dqn_tripinfo_final.xml"
)

obs, _ = env_eval.reset()
done = False
datos_recolectados = []
print("Ejecutando evaluacion...")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    datos_recolectados.append(info)
    
    if env_eval.unwrapped.sumo.simulation.getMinExpectedNumber() == 0:
        print(f"Evaluacion completada. Red vaciada en {env_eval.unwrapped.sim_step}s.")
        break
    done = terminated or truncated

env_eval.close()

df_metricas = pd.DataFrame(datos_recolectados)
ruta_csv = f"{OUT_FOLDER}/dqn_dataset_sumo.csv"
df_metricas.to_csv(ruta_csv, index=False)
print(f"CSV exportado: {ruta_csv}")
print(f"XML generado: {OUT_FOLDER}/dqn_tripinfo_final.xml")
```

### Celda 8: Verificacion CSV
```python
df_metricas = pd.read_csv(f"{OUT_FOLDER}/dqn_dataset_sumo.csv")
print(f"Registros cargados: {len(df_metricas)}")
```

### Celda 9: Dashboard (3 subplots)
Identico a Exp3: waitingTime scatter + media movil, timeLoss scatter + media movil, cola plot + fill_between.

### Celda 10: Reporte de metricas
Identico a Exp3: total vehiculos, velocidad, espera, perdido, cola.

## Archivos generados

### Entrenamiento (solo modelos)
- `resultados_dqn/last_model.zip`
- `resultados_dqn/best_model.zip`

### Validacion (unicos exports de datos)
- `resultados_dqn/dqn_tripinfo_final.xml`
- `resultados_dqn/dqn_dataset_sumo.csv`
- `resultados_dqn/evaluacion_final_conn1_ep1.csv` (auto por sumo-rl)

## Verificacion
1. Ejecutar notebook completo en Jupyter
2. Verificar que `best_model.zip` y `last_model.zip` existen en `resultados_dqn/`
3. Verificar que `dqn_tripinfo_final.xml` existe y tiene datos de vehiculos
4. Verificar que `dqn_dataset_sumo.csv` tiene columnas `step`, `system_total_stopped`, etc.
5. Verificar que el dashboard se genera correctamente
6. Verificar que el reporte de metricas muestra valores coherentes
