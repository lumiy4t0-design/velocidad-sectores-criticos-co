import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página web
st.set_page_config(page_title="Predicción de Riesgo Vial", page_icon="🚨", layout="wide")
st.title("🚨 Clasificador de Sectores Críticos por Exceso de Velocidad")
st.subheader("Proyecto Integrador - Fundamentos de Ciencia de Datos")
st.write("Aplicación para predecir si un tramo vial en Colombia es de **Alto Riesgo** mediante Machine Learning.")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_siniestralidad_vial.pkl')
    scaler = joblib.load('escalador_vial.pkl')
    columnas = joblib.load('columnas_train.pkl')
    cols_num = joblib.load('cols_num.pkl')
    return modelo, scaler, columnas, cols_num

try:
    modelo, scaler, columnas_train, cols_num = cargar_modelo()
    st.sidebar.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error("❌ No se encontraron los archivos .pkl en la carpeta.")
    st.exception(e)
    st.stop()

# Entradas del usuario en la barra lateral
st.sidebar.header("Parámetros del Tramo Vial")
latitud = st.sidebar.number_input("Latitud", value=4.570868, format="%.6f")
longitud = st.sidebar.number_input("Longitud", value=-74.297333, format="%.6f")

if st.sidebar.button("🔍 Evaluar Riesgo"):
    # Construir DataFrame de entrada con las mismas columnas usadas en entrenamiento
    entrada = pd.DataFrame(0, index=[0], columns=columnas_train)

    # Asignar solo las features que el usuario ingresa y existen en el modelo
    if 'Latitud' in entrada.columns:
        entrada['Latitud'] = latitud
    if 'Longitud' in entrada.columns:
        entrada['Longitud'] = longitud

    # Detectar si el modelo es Regresión Logística (requiere escalado)
    es_logistica = modelo.__class__.__name__ == 'LogisticRegression'

    if es_logistica:
        entrada_escalada = entrada.copy()
        cols_validas = [c for c in cols_num if c in entrada_escalada.columns]
        # Transformar SOLO las columnas numéricas que existen
        entrada_escalada[cols_validas] = scaler.transform(entrada_escalada[cols_validas])
        X_input = entrada_escalada
    else:
        X_input = entrada

    # Predicción
    prediccion = modelo.predict(X_input)

    # Probabilidad (usando la clase positiva = 1)
    proba = None
    if hasattr(modelo, 'predict_proba'):
        proba = modelo.predict_proba(X_input)[0][1]

    # Mostrar resultado
    if prediccion[0] == 1:
        st.error("🚨 **ALTO RIESGO** detectado en el tramo.")
    else:
        st.success("✅ **BAJO RIESGO** en el tramo.")

    if proba is not None:
        st.metric("Probabilidad de Alto Riesgo", f"{proba:.2%}")

    st.subheader("Datos evaluados")
    cols_mostrar = [c for c in ['Latitud', 'Longitud'] if c in entrada.columns]
    st.dataframe(entrada[cols_mostrar] if cols_mostrar else entrada)

