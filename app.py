import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Siniestralidad Vial Colombia",
    page_icon="🚨",
    layout="wide"
)

# Título y encabezado
st.title("🚨 Clasificador de Sectores Críticos por Exceso de Velocidad")
st.markdown(
    "**Proyecto Integrador - Fundamentos de Ciencia de Datos** | "
    "*Agencia Nacional de Seguridad Vial (Colombia)*"
)
st.markdown("---")

# Cargar artefactos
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

# Sidebar: Formulario de entrada
st.sidebar.header("📍 Parámetros del Tramo Vial")

departamentos = [
    'VALLE DEL CAUCA', 'ANTIOQUIA', 'CUNDINAMARCA', 'ATLANTICO', 'SANTANDER',
    'BOLIVAR', 'BOYACA', 'CALDAS', 'CAUCA', 'CESAR', 'CORDOBA', 'HUILA',
    'MAGDALENA', 'NARIÑO', 'NORTE DE SANTANDER', 'QUINDIO', 'RISARALDA',
    'TOLIMA', 'OTRO'
]
entidades = ['INVIAS', 'ANI', 'ALCALDIA', 'GOBERNACION', 'OTRA']
tipos_calzada = ['SENCILLA', 'DOBLE', 'MULTICARRIL', 'OTRO']

dep_seleccionado = st.sidebar.selectbox("Seleccione el Departamento:", departamentos)
entidad_seleccionada = st.sidebar.selectbox("Entidad responsable del tramo:", entidades)
tipo_calzada = st.sidebar.selectbox("Tipo de Calzada:", tipos_calzada)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Coordenadas del Tramo en Colombia")

latitud = st.sidebar.slider(
    "Latitud (Norte):", min_value=1.0, max_value=12.5, value=4.57, step=0.01
)
longitud = st.sidebar.slider(
    "Longitud (Oeste):", min_value=-79.0, max_value=-66.0, value=-74.30, step=0.01
)

# Botón de predicción
if st.button("🔍 Evaluar Riesgo del Sector Vial", type="primary"):
    # Crear vector de características de 0s con la estructura de entrenamiento
    input_data = pd.DataFrame(0, index=[0], columns=columnas_train)

    # Asignar valores numéricos
    input_data['Latitud'] = latitud
    input_data['Longitud'] = longitud

    # Normalizar numéricas (solo las que existan en el DataFrame)
    cols_num_presentes = [c for c in cols_num if c in input_data.columns]
    if cols_num_presentes:
        input_data[cols_num_presentes] = scaler.transform(input_data[cols_num_presentes])

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

    # Inferencia
    prediccion = modelo.predict(input_data)[0]
    probabilidades = modelo.predict_proba(input_data)[0]
    prob_alto_riesgo = probabilidades[1] * 100

    st.markdown("### 📊 Resultado de la Evaluación")
    col1, col2 = st.columns(2)

    with col1:
        if prediccion == 1:
            st.error("🚨 **CLASIFICACIÓN: SECTOR DE ALTO RIESGO**")
            st.write(
                "Este tramo vial presenta características asociadas a alta "
                "frecuencia de siniestros por exceso de velocidad."
            )
        else:
            st.success("✅ **CLASIFICACIÓN: SECTOR DE BAJO / MODERADO RIESGO**")
            st.write(
                "Este tramo vial no muestra indicadores de concentración crítica "
                "de exceso de velocidad."
            )

    with col2:
        st.metric(
            label="Probabilidad Estimada de Alto Riesgo",
            value=f"{prob_alto_riesgo:.1f}%"
        )
        st.progress(int(prob_alto_riesgo))

    st.markdown("---")
    st.info(
        f"📌 **Resumen de consulta:** Tramo ubicado en **{dep_seleccionado}** "
        f"(Lat: {latitud}, Lon: {longitud}) a cargo de **{entidad_seleccionada}**."
    )
