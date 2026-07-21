# TrafficFlow Dashboard

Dashboard interactivo para visualizar y comparar los resultados de las simulaciones de control semafórico en la intersección de Loja (Av. Isidro Ayora x 8 de Diciembre).

## Características

- **Métricas comparativas** de los 3 escenarios: Tiempos Fijos, Q-Learning y DQN
- **Gráficos interactivos** con Plotly (evolución temporal, distribución por tipo, patrón de demanda)
- **Análisis detallado** por tipo de vehículo (car, moto, bus)
- **KPI cards** con mejoras porcentuales vs línea base
- **Tablas comparativas** con resaltado automático de mejores valores

## Ejecución

Ejecutar el script `run.sh` desde la carpeta del dashboard:
```bash
cd /home/santiago/Documentos/Tesis/dashboard
./run.sh
```

O directamente sin script:
```bash
/home/santiago/Documentos/Tesis/.venv/bin/python3 -m streamlit run /home/santiago/Documentos/Tesis/dashboard/app.py --server.headless true
```

El dashboard estará disponible en: `http://localhost:8501`

## Estructura del Proyecto

```
dashboard/
  app.py          # Aplicación principal Streamlit
  utils.py        # Funciones de carga y procesamiento de datos
  run.sh          # Script de inicio
  requirements.txt
  README.md
```

## Fuentes de Datos

- **CSV métricas:** `metricas.csv` en cada carpeta de resultados
- **XML tripinfo:** `tripinfo.xml` con datos por vehículo
- **Modelos entrenados:** `qlearning.pkl` y `best_model.zip` (DQN)

## Diseño

Basado en el design system **Data-Dense Dashboard** de UI-UX Pro Max:
- Colores: Primary #1E40AF, Secondary #3B82F6, Accent #D97706
- Fuentes: Fira Code (títulos), Fira Sans (cuerpo)
- Estilo: KPI cards, gráficos interactivos, tablas con highlight
