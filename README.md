
# 🚗 Proyecto Integrador - Siniestralidad Vial en Colombia 🇨🇴

## 📌 Descripción del Proyecto

Este proyecto analiza los sectores críticos por exceso de velocidad en Colombia utilizando datos oficiales de **datos.gov.co** (Agencia Nacional de Seguridad Vial - ANSV).

## 🛠️ Metodología (CRISP-DM)

Se compararon 5 modelos supervisados de clasificación:

- Regresión Logística
- Árbol de Decisión (**Modelo Ganador - F1-Score: 57.14%**)
- Random Forest
- XGBoost
- LightGBM

## 📁 Archivos del Repositorio

- `Proy_Int_FCD_AR.ipynb`: Cuaderno con el proceso completo de limpieza y modelado.
- `app.py`: Aplicación interactiva en Streamlit.
- `requirements.txt`: Librerías requeridas.
- `modelo_siniestralidad_vial.pkl`: Modelo clasificador ganador exportado.
- `escalador_vial.pkl`: Escalador estadístico (StandardScaler) exportado.
- `columnas_train.pkl`: Orden exacto de columnas usadas en entrenamiento.
- `cols_num.pkl`: Lista de columnas numéricas escaladas.
- `categorias_alto_riesgo.pkl`: Categorías usadas para binarizar la variable objetivo.

## 🚀 Cómo ejecutar la app

```bash
pip install -r requirements.txt
streamlit run app.py
