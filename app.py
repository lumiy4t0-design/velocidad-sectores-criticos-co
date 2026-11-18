import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página web
st.set_page_config(
    page_title="Siniestralidad Vial Colombia",
    page_icon="🚨",
    layout="wide"
)

# Título y encabezado principal
st.title("🚨 Clasificador de Sectores Críticos por Exceso de Velocidad")
st.markdown(
    "**Proyecto Integrador - Fundamentos de Ciencia de Datos** | "
    "*Agencia Nacional de Seguridad Vial (Colombia)*"
)
st.markdown("---")

# Cargar artefactos .pkl
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_siniestralidad_vial.pkl')
    scaler = joblib.load('escalador_vial.pkl')
    columnas_train = joblib.load('columnas_train.pkl')
    cols_num = joblib.load('cols_num.pkl')
    return modelo, scaler, columnas_train, cols_num

try:
    modelo, scaler, columnas_train, cols_num = cargar_artefactos()
    st.sidebar.success("✅ Modelo y artefactos cargados correctamente")
except Exception as e:
    st.error("⚠️ Error al cargar los artefactos .pkl. Verifica que estén en el repositorio.")
    st.exception(e)
    st.stop()

# Coordenadas representativas por Departamento en Colombia
coordenadas_departamentos = {
    'VALLE DEL CAUCA': {'lat': 3.4516, 'lon': -76.5320},
    'NARIÑO': {'lat': 1.2136, 'lon': -77.2811},
    'ANTIOQUIA': {'lat': 6.2442, 'lon': -75.5812},
    'CUNDINAMARCA': {'lat': 4.6097, 'lon': -74.0817},
    'ATLANTICO': {'lat': 10.9685, 'lon': -74.7813},
    'SANTANDER': {'lat': 7.1254, 'lon': -73.1198},
    'BOLIVAR': {'lat': 10.3997, 'lon': -75.5144},
    'BOYACA': {'lat': 5.5353, 'lon': -73.3678},
    'CALDAS': {'lat': 5.0689, 'lon': -75.5174},
    'CAUCA': {'lat': 2.4448, 'lon': -76.6147},
    'CESAR': {'lat': 10.4631, 'lon': -73.2532},
    'CORDOBA': {'lat': 8.7480, 'lon': -75.8814},
    'HUILA': {'lat': 2.9273, 'lon': -75.2819},
    'MAGDALENA': {'lat': 11.2408, 'lon': -74.1990},
    'NORTE DE SANTANDER': {'lat': 7.8939, 'lon': -72.5078},
    'QUINDIO': {'lat': 4.5339, 'lon': -75.6811},
    'RISARALDA': {'lat': 4.8133, 'lon': -75.6961},
    'TOLIMA': {'lat': 4.4389, 'lon': -75.2322},
    'OTRO': {'lat': 4.5709, 'lon': -74.2973},
}

# Sidebar: Formulario
st.sidebar.header("📍 Parámetros del Tramo Vial")

dep_seleccionado = st.sidebar.selectbox(
    "Seleccione el Departamento:",
    list(coordenadas_departamentos.keys())
)

entidades = ['INVIAS', 'ANI', 'ALCALDIA', 'GOBERNACION', 'OTRA']
entidad_seleccionada = st.sidebar.selectbox("Entidad responsable:", entidades)

tipos_calzada = ['SENCILLA', 'DOBLE', 'MULTICARRIL', 'OTRO']
tipo_calzada = st.sidebar.selectbox("Tipo de Calzada:", tipos_calzada)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Ajuste de Coordenadas (Opcional)")

coords_def = coordenadas_departamentos.get(dep_seleccionado, {'lat': 4.5709, 'lon': -74.2973})
latitud = st.sidebar.number_input("Latitud (°N):", value=float(coords_def['lat']), format="%.4f")
longitud = st.sidebar.number_input("Longitud (°W):", value=float(coords_def['lon']), format="%.4f")

# Botón de predicción
if st.sidebar.button("🔍 Evaluar Riesgo del Sector Vial", type="primary"):

    # Crear dataframe con ceros según columnas de entrenamiento
    # CORRECCIÓN: index=[0] en lugar de index=
    input_data = pd.DataFrame(0, index=[0], columns=columnas_train)

    # Asignar variables numéricas
    if 'Latitud' in input_data.columns:
        input_data['Latitud'] = latitud
    if 'Longitud' in input_data.columns:
        input_data['Longitud'] = longitud

    # Activar One-Hot Encoding según selección
    col_dep = f"Departamento_{dep_seleccionado}"
    if col_dep in input_data.columns:
        input_data[col_dep] = 1

    col_ent = f"ENTIDAD_{entidad_seleccionada}"
    if col_ent in input_data.columns:
        input_data[col_ent] = 1

    col_calz = f"TIPO_CALZ_{tipo_calzada}"
    if col_calz in input_data.columns:
        input_data[col_calz] = 1

    # REGLA CLAVE: Solo escalar si el modelo es Regresión Logística.
    # El Árbol de Decisión fue entrenado en Colab con datos reales sin escalar (X_train).
    es_logistica = modelo.__class__.__name__ == 'LogisticRegression'

    if es_logistica:
        input_data_eval = input_data.copy()
        num_cols = [c for c in cols_num if c in input_data_eval.columns]
        if num_cols:
            input_data_eval[num_cols] = scaler.transform(input_data_eval[num_cols])
    else:
        input_data_eval = input_data

    # Inferencia
    prediccion = modelo.predict(input_data_eval)[0]
    probabilidades = modelo.predict_proba(input_data_eval)[0]
    prob_alto_riesgo = probabilidades[1] * 100

    # Bloque de Resultados
    st.markdown("### 📊 Resultado de la Evaluación del Sector")
    col1, col2 = st.columns(2)

    with col1:
        if prediccion == 1:
            st.error("🚨 **CLASIFICACIÓN: SECTOR DE ALTO RIESGO**")
            st.write(
                "Este tramo presenta alta vulnerabilidad y concentración "
                "histórica de exceso de velocidad."
            )
        else:
            st.success("✅ **CLASIFICACIÓN: SECTOR DE BAJO / MODERADO RIESGO**")
            st.write(
                "Este tramo muestra un nivel controlado de riesgo vial según "
                "los indicadores analizados."
            )

    with col2:
        st.metric(
            label="Probabilidad Estimada de Alto Riesgo",
            value=f"{prob_alto_riesgo:.1f}%"
        )
        st.progress(int(min(100, max(0, prob_alto_riesgo))))

    st.markdown("---")

    # RESULTADOS GEOGRÁFICOS
    st.markdown("### 🌐 Coordenadas Geográficas Evaluadas")
    col_lat, col_lon, col_dep_info = st.columns(3)
    col_lat.metric("Latitud (Norte)", f"{latitud:.4f}°")
    col_lon.metric("Longitud (Oeste)", f"{longitud:.4f}°")
    col_dep_info.metric("Departamento", dep_seleccionado)

    st.info(
        f"📌 **Resumen Institucional:** Tramo evaluado en **{dep_seleccionado}** "
        f"a cargo de **{entidad_seleccionada}** (Calzada {tipo_calzada})."
    )

else:
    st.info(
        "👈 Seleccione los **Parámetros del Tramo Vial** en el menú de la "
        "izquierda y presione **'Evaluar Riesgo del Sector Vial'**."
    )
