import pandas as pd
import xml.etree.ElementTree as ET
import os
import tempfile
from typing import Dict, List, Tuple, Optional


# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = {
    "Tiempos Fijos": os.path.join(BASE_DIR, "Simulaciones_tiempos_fijos", "resultados"),
    "Q-Learning": os.path.join(BASE_DIR, "Q-Learning", "resultados"),
    "DQN": os.path.join(BASE_DIR, "DQN", "resultados"),
}


def load_csv(scenario: str, custom_path: str = None) -> pd.DataFrame:
    """Carga y normaliza las métricas CSV de un escenario.

    Args:
        scenario: Nombre del escenario (Tiempos Fijos, Q-Learning, DQN)
        custom_path: Ruta personalizada al archivo CSV (opcional)
    """
    if custom_path and os.path.exists(custom_path):
        csv_path = custom_path
    else:
        csv_path = os.path.join(RESULTS[scenario], "metricas.csv")

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalizar nombres de columnas entre escenarios
    if "Paso_Tiempo" in df.columns:
        df = df.rename(columns={
            "Paso_Tiempo": "step",
            "Velocidad_Promedio": "system_mean_speed",
        })
        if "Cola_Actual" in df.columns:
            df["system_total_stopped"] = df["Cola_Actual"]
        elif "Vehiculos_Activos" in df.columns:
            df["system_total_stopped"] = df["Vehiculos_Activos"]
        if "TiempoEspera_Promedio" in df.columns:
            df["system_mean_waiting_time"] = df["TiempoEspera_Promedio"]

    for col in ["step", "system_total_stopped", "system_mean_waiting_time", "system_mean_speed"]:
        if col not in df.columns:
            df[col] = 0

    return df


def load_tripinfo(scenario: str, custom_path: str = None) -> pd.DataFrame:
    """Parsea tripinfo.xml en un DataFrame con métricas por vehículo.

    Args:
        scenario: Nombre del escenario (Tiempos Fijos, Q-Learning, DQN)
        custom_path: Ruta personalizada al archivo XML (opcional)
    """
    if custom_path and os.path.exists(custom_path):
        xml_path = custom_path
    else:
        xml_path = os.path.join(RESULTS[scenario], "tripinfo.xml")

    if not os.path.exists(xml_path):
        return pd.DataFrame()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for trip in root.iter("tripinfo"):
        rows.append({
            "id": trip.get("id"),
            "depart": float(trip.get("depart", 0)),
            "arrival": float(trip.get("arrival", 0)),
            "duration": float(trip.get("duration", 0)),
            "routeLength": float(trip.get("routeLength", 0)),
            "waitingTime": float(trip.get("waitingTime", 0)),
            "waitingCount": int(trip.get("waitingCount", 0)),
            "stopTime": float(trip.get("stopTime", 0)),
            "timeLoss": float(trip.get("timeLoss", 0)),
            "vType": trip.get("vType", "car"),
            "departLane": trip.get("departLane", ""),
            "arrivalLane": trip.get("arrivalLane", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["velocidad_kmh"] = (df["routeLength"] / df["duration"].replace(0, 1)) * 3.6
    return df


def compute_summary(scenario: str, custom_paths: Dict[str, str] = None) -> Dict:
    """Calcula métricas resumen para un escenario.

    Args:
        scenario: Nombre del escenario
        custom_paths: Diccionario con rutas personalizadas {'csv': path, 'xml': path}
    """
    csv_path = custom_paths.get("csv") if custom_paths else None
    xml_path = custom_paths.get("xml") if custom_paths else None

    df_csv = load_csv(scenario, csv_path)
    df_trip = load_tripinfo(scenario, xml_path)

    if df_trip.empty:
        return {
            "total_vehiculos": 0,
            "velocidad_promedio": 0,
            "tiempo_espera_promedio": 0,
            "tiempo_espera_max": 0,
            "tiempo_perdido_promedio": 0,
            "tiempo_perdido_max": 0,
            "cola_promedio": 0,
            "cola_maxima": 0,
        }

    return {
        "total_vehiculos": len(df_trip),
        "velocidad_promedio": round(df_trip["velocidad_kmh"].mean(), 2),
        "tiempo_espera_promedio": round(df_trip["waitingTime"].mean(), 2),
        "tiempo_espera_max": round(df_trip["waitingTime"].max(), 2),
        "tiempo_perdido_promedio": round(df_trip["timeLoss"].mean(), 2),
        "tiempo_perdido_max": round(df_trip["timeLoss"].max(), 2),
        "cola_promedio": round(df_csv["system_total_stopped"].mean(), 2) if not df_csv.empty and "system_total_stopped" in df_csv.columns else 0,
        "cola_maxima": int(df_csv["system_total_stopped"].max()) if not df_csv.empty and "system_total_stopped" in df_csv.columns else 0,
    }


def get_all_summaries(custom_paths: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
    """Retorna un DataFrame comparativo de todos los escenarios.

    Args:
        custom_paths: Diccionario de rutas personalizadas por escenario
                      Ejemplo: {'Q-Learning': {'csv': '/path/to/metricas.csv', 'xml': '/path/to/tripinfo.xml'}}
    """
    summaries = []
    for scenario in RESULTS.keys():
        paths = custom_paths.get(scenario) if custom_paths else None
        s = compute_summary(scenario, paths)
        s["escenario"] = scenario
        summaries.append(s)
    df = pd.DataFrame(summaries)
    return df.set_index("escenario") if not df.empty else df


def get_vehicle_type_breakdown(scenario: str, custom_paths: Dict[str, str] = None) -> pd.DataFrame:
    """Retorna la distribución por tipo de vehículo de un escenario.

    Args:
        scenario: Nombre del escenario
        custom_paths: Diccionario con rutas personalizadas {'csv': path, 'xml': path}
    """
    xml_path = custom_paths.get("xml") if custom_paths else None
    df = load_tripinfo(scenario, xml_path)
    if df.empty:
        return pd.DataFrame()
    return df.groupby("vType").agg(
        cantidad=("id", "count"),
        espera_promedio=("waitingTime", "mean"),
        duracion_promedio=("duration", "mean"),
    ).round(2)


def get_time_bins(scenario: str, bin_size: int = 300, custom_paths: Dict[str, str] = None) -> pd.DataFrame:
    """Agrupa vehículos por tiempo de llegada para mostrar el patrón de demanda.

    Args:
        scenario: Nombre del escenario
        bin_size: Tamaño del intervalo en segundos (default: 300 = 5 minutos)
        custom_paths: Diccionario con rutas personalizadas {'csv': path, 'xml': path}
    """
    xml_path = custom_paths.get("xml") if custom_paths else None
    df = load_tripinfo(scenario, xml_path)
    if df.empty:
        return pd.DataFrame()

    max_time = int(df["depart"].max()) + bin_size
    bins = list(range(0, max_time, bin_size))
    labels = [f"{i//60}-{(i+bin_size)//60}min" for i in bins[:-1]]

    df["intervalo"] = pd.cut(df["depart"], bins=bins, labels=labels, right=False)
    result = df.groupby("intervalo", observed=False).agg(
        vehiculos=("id", "count"),
        espera_promedio=("waitingTime", "mean"),
    ).reset_index()

    return result
