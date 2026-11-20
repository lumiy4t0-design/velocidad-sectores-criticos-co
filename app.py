import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página web
st.set_page_config(page_title="Predicción de Riesgo Vial - Colombia", page_icon="🚨", layout="wide")
st.title("🚨 Clasificador de Sectores Críticos por Exceso de Velocidad")
st.subheader("Proyecto Integrador - Fundamentos de Ciencia de Datos (ANSV Colombia)")
st.write("Aplicación para evaluar el nivel de riesgo en las carreteras colombianas mediante Machine Learning.")

# Cargar el modelo entrenado y sus artefactos
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_siniestralidad_vial.pkl')
    scaler = joblib.load('escalador_vial.pkl')
    columnas = joblib.load('columnas_train.pkl')
    cols_num = joblib.load('cols_num.pkl')
    return modelo, scaler, columnas, cols_num

try:
    modelo, scaler, columnas_train, cols_num = cargar_modelo()
    st.sidebar.success("✅ Modelo y artefactos (.pkl) cargados")
except Exception as e:
    st.error("❌ No se encontraron los archivos .pkl en el directorio.")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------------------------
# 1. TOP DE INICIO: TABLA DE SECTORES DE MAYOR RIESGO EN COLOMBIA
# ------------------------------------------------------------------------------
st.markdown("### 🏆 Top Sectores Críticos de Siniestralidad Vial (ANSV)")
st.write("Principales puntos geográficos con mayor registro de riesgo por exceso de velocidad:")

top_sectores = pd.DataFrame([
    {"Departamento": "VALLE DEL CAUCA", "Municipio": "CALI", "Sector / Tramo": "Autopista Sur", "Latitud": 3.451600, "Longitud": -76.532000, "Riesgo Histórico": "Alto"},
    {"Departamento": "NARIÑO", "Municipio": "PASTO", "Sector / Tramo": "RumPast_Tramo_UF5", "Latitud": 1.163092, "Longitud": -77.286501, "Riesgo Histórico": "Muy Alto"},
    {"Departamento": "SANTANDER", "Municipio": "BUCARAMANGA", "Sector / Tramo": "Vía Palonegro", "Latitud": 7.119349, "Longitud": -73.122742, "Riesgo Histórico": "Alto"},
    {"Departamento": "RISARALDA", "Municipio": "PEREIRA", "Sector / Tramo": "Variante Romelia", "Latitud": 4.813330, "Longitud": -75.696110, "Riesgo Histórico": "Alto"},
    {"Departamento": "NORTE DE SANTANDER", "Municipio": "CÚCUTA", "Sector / Tramo": "Anillo Vial Oriental", "Latitud": 7.893910, "Longitud": -72.507820, "Riesgo Histórico": "Alto"},
    {"Departamento": "META", "Municipio": "VILLAVICENCIO", "Sector / Tramo": "Vía al Llano UF3", "Latitud": 4.142000, "Longitud": -73.626640, "Riesgo Histórico": "Alto"},
    {"Departamento": "MAGDALENA", "Municipio": "SANTA MARTA", "Sector / Tramo": "Vía Ciénaga - Sta Marta", "Latitud": 11.240790, "Longitud": -74.199040, "Riesgo Histórico": "Alto"},
])
st.dataframe(top_sectores, use_container_width=True)

# ------------------------------------------------------------------------------
# 2. MODO DE EVALUACIÓN INTERACTIVA
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Evaluación Interactiva de un Tramo Vial")
modo_busqueda = st.radio(
    "¿Cómo deseas seleccionar el tramo vial a evaluar?",
    ["📌 Seleccionar de la lista de sectores críticos", "🌐 Ingresar coordenadas manualmente"],
    horizontal=True,
)

latitud_input = 4.570868
longitud_input = -74.297333

if modo_busqueda == "📌 Seleccionar de la lista de sectores críticos":
    opciones = [
        f"{row['Departamento']} - {row['Municipio']} ({row['Sector / Tramo']})"
        for _, row in top_sectores.iterrows()
    ]
    seleccion = st.selectbox("Elige un sector para cargar sus coordenadas:", opciones)
    idx_sel = opciones.index(seleccion)
    latitud_input = float(top_sectores.iloc[idx_sel]['Latitud'])
    longitud_input = float(top_sectores.iloc[idx_sel]['Longitud'])
    st.info(
        f"📍 Coordenadas cargadas automáticamente ➔ "
        f"**Latitud:** `{latitud_input}` | **Longitud:** `{longitud_input}`"
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        latitud_input = st.number_input("Latitud (Norte)", value=4.570868, format="%.6f")
    with col2:
        longitud_input = st.number_input("Longitud (Oeste)", value=-74.297333, format="%.6f")

# ------------------------------------------------------------------------------
# 3. PREDICCIÓN CON EL MODELO (.PKL)
# ------------------------------------------------------------------------------
if st.button("🚨 Evaluar Riesgo del Sector", type="primary"):

    # Reconstruir el vector de entrada con las columnas exactas del modelo
    entrada = pd.DataFrame(0, index=[0], columns=columnas_train)

    if 'Latitud' in entrada.columns:
        entrada['Latitud'] = latitud_input
    if 'Longitud' in entrada.columns:
        entrada['Longitud'] = longitud_input

    # Escalado condicional (solo si el modelo lo requiere y las columnas coinciden)
    es_logistica = modelo.__class__.__name__ == 'LogisticRegression'
    if es_logistica:
        entrada_escalada = entrada.copy()
        cols_validas = [c for c in cols_num if c in entrada_escalada.columns]
        if cols_validas:
            entrada_escalada[cols_validas] = scaler.transform(entrada_escalada[cols_validas])
        X_input = entrada_escalada
    else:
        X_input = entrada

    # Inferencia
    try:
        prediccion = modelo.predict(X_input)
        proba = None
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba(X_input)[0][1]
    except Exception as e:
        st.error("❌ Error al ejecutar la predicción con el modelo.")
        st.exception(e)
        st.stop()

    # Resultados en pantalla
    st.markdown("#### 📊 Resultado del Análisis por Inteligencia Artificial")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if int(prediccion[0]) == 1:
            st.error("🚨 **ALTO RIESGO DE SINIESTRALIDAD VIAL**")
            st.warning("⚠️ **Recomendación ANSV:** Instalar fotomultas y reforzar patrullajes policiales en estas coordenadas.")
        else:
            st.success("✅ **SECTOR DE BAJO / MODERADO RIESGO**")
            st.info("ℹ️ Tramo vial con niveles de riesgo dentro de los rangos normales.")

    with col_res2:
        if proba is not None:
            st.metric("Probabilidad Estimada de Alto Riesgo", f"{proba:.1%}")



