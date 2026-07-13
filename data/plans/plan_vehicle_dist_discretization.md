# Plan: Distribucion Vehicular + Discretizacion de Estados + Epsilon + Episodios

## Contexto
Se necesitan 4 cambios en todos los experimentos:
1. Actualizar distribucion de tipos de vehiculo a datos reales (69.58% Ligeros, 8.76% Motos, 21.66% Pesados)
2. Cambiar discretizacion de estados en Q-Learning a esquema del estado del arte: s = (nvehicles, twaiting, lqueue, rarrival)
3. Cambiar epsilon decay a lineal completo (1.0 → 0.0)
4. Ajustar episodios a 30

## Decisiones confirmadas
- Pesados → todo a `bus` (no agregar truck)
- Solo modificar archivos V2 (Weibull V2 y Poisson V2)
- Epsilon: decaimiento lineal completo de 1.0 a 0.0 en 30 episodios
- Exp1 (ql_puro) NO lleva discretizacion (es el baseline continuo)
- Exp2 (ql_discr) y Exp3 (ql_doble) llevan nueva discretizacion

---

## Paso 1: Distribucion vehicular en archivos .rou.xml V2

### Weibull V2 (2418 vehiculos)
| Tipo | Actual | Target |
|------|--------|--------|
| car | 1791 (74.07%) | 1683 (69.58%) |
| moto | 358 (14.81%) | 212 (8.76%) |
| bus | 269 (11.12%) | 523 (21.66%) |

### Poisson V2 (1179 vehiculos)
| Tipo | Actual | Target |
|------|--------|--------|
| car | 869 (73.71%) | 820 (69.58%) |
| moto | 172 (14.59%) | 103 (8.76%) |
| bus | 138 (11.70%) | 256 (21.66%) |

### Implementacion
Script Python que:
1. Parsea el XML con `xml.etree.ElementTree`
2. Extrae todos los elementos `<vehicle>`
3. Calcula conteos target (redondeo para que sumen total)
4. Reasigna `type` attribute: primero excess de car/moto → bus
5. Guarda el XML modificado

Archivos a modificar:
- `Distribuciones/Dist_Weibull_V2/loja_intersection_weibull.rou.xml`
- `Distribuciones/Dist_Poisson_V2/loja_intersection_poisson.rou.xml`

---

## Paso 2: Nueva discretizacion de estados (Exp2 y Exp3)

### Esquema del estado del arte
```
s = (nvehicles, twaiting, lqueue, rarrival)
```

| Componente | Descripcion | Bins |
|------------|-------------|------|
| nvehicles | Vehiculos en red | 6 bins: 0-5, 6-10, 11-15, 16-20, 21-25, 26+ |
| twaiting | Tiempo espera total (s) | 6 bins: 0-3, 4-7, 8-12, 13-20, 21-30, >30 |
| lqueue | Longitud de cola | 4 bins: 0-3, 4-7, 8-12, 13+ |
| rarrival | Tasa de llegada (veh/step) | 3 bins: 0-1, 2-3, 4+ |

### Tamaño espacio de estados
6 × 6 × 4 × 3 = **432 estados** (manejable para Q-Learning)

### Fuentes de datos (confirmado via TraCI)
| Metrica | Fuente |
|---------|--------|
| nvehicles | `env.unwrapped.sumo.vehicle.getIDCount()` |
| twaiting | `info['system_total_waiting_time']` |
| lqueue | `info['system_total_stopped']` |
| rarrival | Delta de `getIDCount()` entre steps (tracking manual) |

### Implementacion
Reemplazar `DiscretizedObservationWrapper` en Exp2 y Exp3 con `TrafficStateWrapper`:

```python
class TrafficStateWrapper(gym.Wrapper):
    """Estado: (nvehicles, twaiting, lqueue, rarrival) con bins del estado del arte."""
    
    def __init__(self, env):
        super().__init__(env)
        self._prev_vehicle_count = 0
        # Nuevo espacio de observacion: 4 valores discretos
        self.observation_space = gym.spaces.MultiDiscrete([6, 6, 4, 3])
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_vehicle_count = self.env.unwrapped.sumo.vehicle.getIDCount()
        return self._compute_state(info), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._compute_state(info)
        return state, reward, terminated, truncated, info
    
    def _compute_state(self, info):
        sumo = self.env.unwrapped.sumo
        
        # 1. nvehicles (6 bins)
        nvehicles = sumo.vehicle.getIDCount()
        nveh_bin = min(nvehicles // 5, 5)  # 0-5→0, 6-10→1, ..., 26+→5
        
        # 2. twaiting (6 bins)
        twaiting = info.get('system_total_waiting_time', 0)
        twait_bin = self._bin_waiting(twaiting)
        
        # 3. lqueue (4 bins)
        lqueue = info.get('system_total_stopped', 0)
        lq_bin = self._bin_queue(lqueue)
        
        # 4. rarrival (3 bins)
        current = sumo.vehicle.getIDCount()
        rarrival = max(0, current - self._prev_vehicle_count)
        self._prev_vehicle_count = current
        ra_bin = min(rarrival // 2, 2)  # 0-1→0, 2-3→1, 4+→2
        
        return (nveh_bin, twait_bin, lq_bin, ra_bin)
    
    @staticmethod
    def _bin_waiting(t):
        if t <= 3: return 0
        if t <= 7: return 1
        if t <= 12: return 2
        if t <= 20: return 3
        if t <= 30: return 4
        return 5
    
    @staticmethod
    def _bin_queue(q):
        if q <= 3: return 0
        if q <= 7: return 1
        if q <= 12: return 2
        return 3
```

### Notebooks a modificar
- `Simulaciones_Q-Learning/ql_discr/qlearning_exp2_discr.ipynb`
- `Simulaciones_Q-Learning/ql_doble/qlearning_exp3_rec_doble.ipynb`

### Notebooks NO modificados
- `Simulaciones_Q-Learning/ql_puro/qlearning_exp1_no_discret.ipynb` (baseline continuo)
- `Simulaciones_DQN/dqn_doble/dqn_weibull.ipynb` (DQN usa estado continuo)

---

## Paso 3: Epsilon decay lineal completo (3 notebooks Q-Learning)

### Cambio
Formula del estado del arte: `epsilon = 1 - (Episodios_Actuales / Episodios_Totales)`

```python
# ANTES (multiplicativo fijo, independiente de epocas):
exploracion.epsilon = max(exploracion.epsilon * 0.90, exploracion.min_epsilon)

# DESPUES (lineal completo, de 1.0 a 0.0):
exploracion.epsilon = 1.0 - (episode / args.episodes)
```

### Tabla de decaimiento (30 episodios, lineal completo)
| Ep | Epsilon |
|----|---------|
| 1 | 0.967 |
| 5 | 0.833 |
| 10 | 0.667 |
| 15 | 0.500 |
| 20 | 0.333 |
| 25 | 0.167 |
| 30 | 0.000 |

### Notebooks a modificar
- `ql_puro/qlearning_exp1_no_discret.ipynb`
- `ql_discr/qlearning_exp2_discr.ipynb`
- `ql_doble/qlearning_exp3_rec_doble.ipynb`

---

## Paso 4: Episodios a 30 (3 notebooks Q-Learning)

### Cambio
```python
# ANTES:
prs.add_argument("-episodes", dest="episodes", type=int, default=25)

# DESPUES:
prs.add_argument("-episodes", dest="episodes", type=int, default=30)
```

### Notebooks a modificar
- `ql_puro/qlearning_exp1_no_discret.ipynb`
- `ql_discr/qlearning_exp2_discr.ipynb`
- `ql_doble/qlearning_exp3_rec_doble.ipynb`

---

## Resumen de archivos a modificar

| Archivo | Cambios |
|---------|---------|
| `Distribuciones/Dist_Weibull_V2/loja_intersection_weibull.rou.xml` | Distribucion vehicular |
| `Distribuciones/Dist_Poisson_V2/loja_intersection_poisson.rou.xml` | Distribucion vehicular |
| `ql_puro/qlearning_exp1_no_discret.ipynb` | Epsilon lineal + 30 episodios |
| `ql_discr/qlearning_exp2_discr.ipynb` | Nueva discretizacion + Epsilon lineal + 30 episodios |
| `ql_doble/qlearning_exp3_rec_doble.ipynb` | Nueva discretizacion + Epsilon lineal + 30 episodios |

---

## Verificacion

1. **Distribucion vehicular**: Parsear XMLs modificados y verificar conteos = targets
2. **Discretizacion**: Ejecutar 1 episodio de Exp2/Exp3 y verificar que el estado es una tupla de 4 elementos con valores en rangos esperados
3. **Epsilon**: Verificar en output logs que epsilon decae linealmente de 0.967 a 0.0 en 30 episodios
4. **Episodios**: Verificar que el entrenamiento corre exactamente 30 episodios
