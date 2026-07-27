import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import tempfile

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_csv,
    load_tripinfo,
    compute_summary,
    get_all_summaries,
    get_vehicle_type_breakdown,
    get_time_bins,
    RESULTS,
)

# Page config
st.set_page_config(
    page_title="TrafficFlow Dashboard",
    page_icon="traffic_light",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS using design system tokens
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --color-primary: #1E40AF;
        --color-secondary: #3B82F6;
        --color-accent: #D97706;
        --color-background: #F8FAFC;
        --color-foreground: #1E3A8A;
        --color-muted: #E9EEF6;
        --color-border: #DBEAFE;
        --color-destructive: #DC2626;
    }

    .main-header {
        font-family: 'Fira Code', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--color-foreground);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-family: 'Fira Sans', sans-serif;
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.1);
    }

    .metric-label {
        font-family: 'Fira Sans', sans-serif;
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--color-primary);
    }

    .metric-unit {
        font-family: 'Fira Sans', sans-serif;
        font-size: 0.85rem;
        color: #94A3B8;
    }

    .improvement-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-green {
        background: #DCFCE7;
        color: #166534;
    }

    .badge-red {
        background: #FEE2E2;
        color: #991B1B;
    }

    .section-divider {
        border-top: 2px solid var(--color-border);
        margin: 2rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Fira Sans', sans-serif;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background: var(--color-primary);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: float, unit: str, delta: float = None):
    """Render a styled metric card."""
    delta_html = ""
    if delta is not None:
        if delta < 0:
            delta_html = f'<span class="improvement-badge badge-green">{delta:.1f}%</span>'
        elif delta > 0:
            delta_html = f'<span class="improvement-badge badge-red">+{delta:.1f}%</span>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.2f}</div>
        <div class="metric-unit">{unit} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format comparison table with proper names and 2 decimal places."""
    column_names = {
        "total_vehiculos": "Total Vehículos",
        "velocidad_promedio": "Velocidad Prom. (km/h)",
        "tiempo_espera_promedio": "Espera Prom. (s)",
        "tiempo_espera_max": "Espera Máx. (s)",
        "tiempo_perdido_promedio": "Pérdida Prom. (s)",
        "tiempo_perdido_max": "Pérdida Máx. (s)",
        "cola_promedio": "Cola Prom. (veh)",
        "cola_maxima": "Cola Máx. (veh)",
    }

    formatted = df.rename(columns=column_names)

    # Format numbers to 2 decimal places
    for col in formatted.columns:
        if col != "Total Vehículos":
            formatted[col] = formatted[col].apply(lambda x: f"{x:.2f}")
        else:
            formatted[col] = formatted[col].apply(lambda x: f"{int(x)}")

    return formatted


def save_uploaded_file(uploaded_file, scenario: str, file_type: str) -> str:
    """Save uploaded file to a temporary location and return the path."""
    if uploaded_file is not None:
        temp_dir = os.path.join(tempfile.gettempdir(), "trafficflow_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(uploaded_file.name)[1]
        temp_path = os.path.join(temp_dir, f"{scenario}_{file_type}{ext}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return temp_path
    return None


def main():
    # Initialize session state for file persistence
    for s in RESULTS.keys():
        if f"files_{s}" not in st.session_state:
            st.session_state[f"files_{s}"] = {"csv": None, "xml": None}

    # Header
    st.markdown('<div class="main-header">TrafficFlow Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparación de Estrategias de Control Semafórico - Intersección Loja</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Configuración")
        scenario = st.selectbox(
            "Escenario de Simulación",
            list(RESULTS.keys()),
            index=0,
        )

        st.markdown("---")
        st.markdown("### Cargar Datos Personalizados")
        st.markdown(f"*Sube archivos CSV o XML para el escenario: **{scenario}***")

        uploaded_csv = st.file_uploader(
            f"Archivo CSV ({scenario})",
            type=["csv"],
            key=f"csv_{scenario}",
            help="Métricas temporales de la simulación",
        )

        uploaded_xml = st.file_uploader(
            f"Archivo XML - Tripinfo ({scenario})",
            type=["xml"],
            key=f"xml_{scenario}",
            help="Información de viajes por vehículo",
        )

        # Save uploaded files to session state
        if uploaded_csv is not None:
            path = save_uploaded_file(uploaded_csv, scenario, "csv")
            st.session_state[f"files_{scenario}"]["csv"] = path

        if uploaded_xml is not None:
            path = save_uploaded_file(uploaded_xml, scenario, "xml")
            st.session_state[f"files_{scenario}"]["xml"] = path

        # Show loaded files status
        loaded_csv = st.session_state[f"files_{scenario}"]["csv"]
        loaded_xml = st.session_state[f"files_{scenario}"]["xml"]

        if loaded_csv or loaded_xml:
            st.info("Archivos personalizados cargados. Se usarán en lugar de los originales.")
            if loaded_csv:
                st.caption(f"CSV: {os.path.basename(loaded_csv)}")
            if loaded_xml:
                st.caption(f"XML: {os.path.basename(loaded_xml)}")

        # Button to clear files for current scenario
        if loaded_csv or loaded_xml:
            if st.button("Limpiar archivos personalizados", key=f"clear_{scenario}"):
                st.session_state[f"files_{scenario}"] = {"csv": None, "xml": None}
                st.rerun()

        st.markdown("---")
        st.markdown("### Datos del Proyecto")
        st.markdown("""
        - **Intersección:** Av. Isidro Ayora x 8 de Diciembre
        - **Demanda:** 2334 vehículos (Weibull)
        - **Simulación:** SUMO 1.26.0
        - **Ciclo:** 3600 segundos
        """)

    # Get file paths from session state
    csv_path = st.session_state[f"files_{scenario}"]["csv"]
    xml_path = st.session_state[f"files_{scenario}"]["xml"]

    # Build current scenario custom paths (None if no files uploaded)
    current_custom = None
    if csv_path or xml_path:
        current_custom = {"csv": csv_path, "xml": xml_path}

    # Load data for current scenario
    df_csv = load_csv(scenario, csv_path)
    df_trip = load_tripinfo(scenario, xml_path)
    summary = compute_summary(scenario, current_custom)

    # Show data source indicator
    if current_custom:
        sources = []
        if csv_path:
            sources.append("CSV personalizado")
        if xml_path:
            sources.append("XML personalizado")
        st.success(f"Fuente de datos: {', '.join(sources)} para {scenario}")

    # For comparison table: collect all custom paths from all scenarios
    all_custom_paths = {}
    for s in RESULTS.keys():
        s_csv = st.session_state[f"files_{s}"]["csv"]
        s_xml = st.session_state[f"files_{s}"]["xml"]
        if s_csv or s_xml:
            all_custom_paths[s] = {"csv": s_csv, "xml": s_xml}

    all_summaries = get_all_summaries(all_custom_paths)

    # KPI Cards
    st.markdown("### Métricas Clave")

    # Calculate improvements vs baseline for RL scenarios
    baseline = compute_summary("Tiempos Fijos") if scenario != "Tiempos Fijos" else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = None
        if baseline and baseline["total_vehiculos"] > 0:
            delta = ((summary["tiempo_espera_promedio"] - baseline["tiempo_espera_promedio"])
                     / baseline["tiempo_espera_promedio"]) * 100
        render_metric_card(
            "Tiempo Espera Promedio",
            summary["tiempo_espera_promedio"],
            "segundos",
            delta,
        )

    with col2:
        delta = None
        if baseline and baseline["total_vehiculos"] > 0:
            delta = ((summary["cola_promedio"] - baseline["cola_promedio"])
                     / baseline["cola_promedio"]) * 100
        render_metric_card(
            "Cola Promedio",
            summary["cola_promedio"],
            "vehículos",
            delta,
        )

    with col3:
        delta = None
        if baseline and baseline["total_vehiculos"] > 0:
            delta = ((summary["velocidad_promedio"] - baseline["velocidad_promedio"])
                     / baseline["velocidad_promedio"]) * 100
        render_metric_card(
            "Velocidad Promedio",
            summary["velocidad_promedio"],
            "km/h",
            delta,
        )

    with col4:
        render_metric_card(
            "Tiempo Perdido Promedio",
            summary["tiempo_perdido_promedio"],
            "segundos",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Comparison Table
    st.markdown("### Comparación de Escenarios")
    formatted_table = format_comparison_table(all_summaries)
    st.dataframe(
        formatted_table.style.highlight_min(
            subset=["Espera Prom. (s)", "Espera Máx. (s)", "Cola Prom. (veh)",
                    "Pérdida Prom. (s)", "Pérdida Máx. (s)"],
            color="#DCFCE7",
            axis=0,
        ).highlight_max(
            subset=["Velocidad Prom. (km/h)"],
            color="#DCFCE7",
            axis=0,
        ),
        use_container_width=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Charts
    st.markdown("### Análisis Visual")

    tab1, tab2, tab3 = st.tabs(["Evolución Temporal", "Distribución por Tipo", "Patrón de Demanda"])

    with tab1:
        # Grafico 1: Vehiculos en Cola
        st.markdown("**Vehículos en Cola**")
        if not df_csv.empty:
            fig_cola = go.Figure()
            fig_cola.add_trace(
                go.Scatter(
                    x=df_csv["step"],
                    y=df_csv["system_total_stopped"],
                    mode="lines",
                    name="Cola",
                    line=dict(color="#3B82F6", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59, 130, 246, 0.1)",
                )
            )
            fig_cola.update_layout(
                height=350,
                font=dict(family="Fira Sans, sans-serif"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Paso de Simulación",
                yaxis_title="Vehículos",
                showlegend=False,
            )
            st.plotly_chart(fig_cola, use_container_width=True)
        else:
            st.warning("No hay datos de métricas temporales.")

        # Grafico 2: Tiempo de Espera Promedio
        st.markdown("**Tiempo de Espera Promedio**")
        if not df_trip.empty:
            df_trip_sorted = df_trip.sort_values("depart").reset_index(drop=True)
            window_size = max(1, len(df_trip_sorted) // 50)
            df_trip_sorted["waiting_rolling"] = df_trip_sorted["waitingTime"].rolling(
                window=window_size, min_periods=1
            ).mean()

            fig_espera = go.Figure()
            fig_espera.add_trace(
                go.Scatter(
                    x=df_trip_sorted["depart"],
                    y=df_trip_sorted["waitingTime"],
                    mode="markers",
                    name="Espera por vehículo",
                    marker=dict(color="#3B82F6", size=4, opacity=0.4),
                )
            )
            fig_espera.add_trace(
                go.Scatter(
                    x=df_trip_sorted["depart"],
                    y=df_trip_sorted["waiting_rolling"],
                    mode="lines",
                    name="Media móvil",
                    line=dict(color="#D97706", width=3),
                )
            )
            fig_espera.update_layout(
                height=350,
                font=dict(family="Fira Sans, sans-serif"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Segundo de Llegada (Depart)",
                yaxis_title="Tiempo de Espera (s)",
                showlegend=True,
            )
            st.plotly_chart(fig_espera, use_container_width=True)
        else:
            st.warning("No hay datos de viajes (tripinfo).")

    with tab2:
        vtype_df = get_vehicle_type_breakdown(scenario, current_custom)
        if not vtype_df.empty:
            col_a, col_b = st.columns(2)

            with col_a:
                fig_pie = px.pie(
                    vtype_df.reset_index(),
                    values="cantidad",
                    names="vType",
                    title="Distribución por Tipo de Vehículo",
                    color_discrete_sequence=["#3B82F6", "#D97706", "#1E40AF"],
                    hole=0.4,
                )
                fig_pie.update_layout(
                    font=dict(family="Fira Sans, sans-serif"),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                fig_bar = px.bar(
                    vtype_df.reset_index(),
                    x="vType",
                    y="espera_promedio",
                    title="Tiempo de Espera Promedio por Tipo",
                    color="vType",
                    color_discrete_sequence=["#3B82F6", "#D97706", "#1E40AF"],
                    text_auto=".2f",
                )
                fig_bar.update_layout(
                    font=dict(family="Fira Sans, sans-serif"),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis_title="Tipo de Vehículo",
                    yaxis_title="Tiempo de Espera (s)",
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No hay datos de tipos de vehículo para este escenario.")

    with tab3:
        time_df = get_time_bins(scenario, custom_paths=current_custom)
        if not time_df.empty:
            fig_demand = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
            )

            fig_demand.add_trace(
                go.Bar(
                    x=time_df["intervalo"],
                    y=time_df["vehiculos"],
                    name="Vehículos",
                    marker_color="#3B82F6",
                    opacity=0.7,
                ),
                secondary_y=False,
            )

            fig_demand.add_trace(
                go.Scatter(
                    x=time_df["intervalo"],
                    y=time_df["espera_promedio"],
                    name="Espera Promedio",
                    line=dict(color="#D97706", width=3),
                    mode="lines+markers",
                ),
                secondary_y=True,
            )

            fig_demand.update_layout(
                title="Patrón Temporal de Demanda (Weibull)",
                height=400,
                font=dict(family="Fira Sans, sans-serif"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_demand.update_xaxes(title_text="Intervalo de Tiempo")
            fig_demand.update_yaxes(title_text="Número de Vehículos", secondary_y=False)
            fig_demand.update_yaxes(title_text="Tiempo de Espera (s)", secondary_y=True)

            st.plotly_chart(fig_demand, use_container_width=True)
        else:
            st.warning("No hay datos de patrón temporal para este escenario.")

    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-family: 'Fira Sans', sans-serif; font-size: 0.85rem;">
        TrafficFlow Dashboard | Universidad Nacional de Loja | Carrera de Computación | 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
