import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Digital Adoption", layout="centered")

# =============================
# Cargar dataset
# =============================
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()

# =============================
# Cargar modelo y transformadores
# =============================
@st.cache_resource
def load_model():
    try:
        modelo = joblib.load("modelo_lightgbm.pkl")
        encoder = joblib.load("encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        num_cols = joblib.load("num_cols.pkl")
        cat_cols = joblib.load("cat_cols.pkl")
        columns_model = joblib.load("columns.pkl")
        return modelo, encoder, scaler, num_cols, cat_cols, columns_model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None, None, None, None

modelo, encoder, scaler, num_cols, cat_cols, columns_model = load_model()

# =============================
# Función de página de gráficas
# =============================
def page_segmentacion():
    st.title("Segmentación de Clientes por Comportamiento Digital")
    
    opcion = st.slider("Selecciona un gráfico", 1, 5, 1)
    
    if opcion == 1:
        st.subheader("Distribución de adopción digital")
        counts = df["digital_adoption_likelihood"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)
        for i, val in enumerate(counts):
            ax.text(i, val+2, str(val), ha='center')
        st.pyplot(fig)
        
    elif opcion == 2:
        st.subheader("Distribución de género")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='CustGender', palette=['pink','skyblue'], ax=ax)
        st.pyplot(fig)
        
    elif opcion == 3:
        st.subheader("Digital Activity Score")
        df["DigitalActivityScore"] = (
            ((df["DigitalTransactionsCount"] - df["DigitalTransactionsCount"].min()) / 
             (df["DigitalTransactionsCount"].max() - df["DigitalTransactionsCount"].min())) +
            ((df["SpendBalanceRatio"] - df["SpendBalanceRatio"].min()) / 
             (df["SpendBalanceRatio"].max() - df["SpendBalanceRatio"].min())) +
            ((df["CustomerTenureYears"] - df["CustomerTenureYears"].min()) / 
             (df["CustomerTenureYears"].max() - df["CustomerTenureYears"].min()))
        )
        fig, ax = plt.subplots()
        sns.histplot(df["DigitalActivityScore"], bins=30, kde=False, ax=ax)
        st.pyplot(fig)
        
    elif opcion == 4:
        st.subheader("Transacciones digitales vs presenciales")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='DigitalTransactionsCount', y='BranchTransactionsCount', ax=ax)
        st.pyplot(fig)
        
    elif opcion == 5:
        st.subheader("Tipos de tarjeta")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='CreditCardType', ax=ax)
        st.pyplot(fig)

# =============================
# Función de página de predicción amigable
# =============================
def page_modelo():
    st.title("Predicción de Adopción Digital (Versión Amigable)")

    if modelo is None:
        st.warning("Modelo no cargado. No se puede realizar predicción.")
        return
    
    st.markdown("Ingresa solo los valores más importantes. Las demás columnas se completan automáticamente.")
    
    # 🔹 Selección de columnas clave
    key_num_cols = ["TransactionAmount (INR)", "CustomerAge", "DigitalTransactionsCount"]
    key_cat_cols = ["CustGender", "CreditCardType"]

    X_input = {}

    # Entradas numéricas
    for c in key_num_cols:
        X_input[c] = st.number_input(f"{c}", value=0.0)
    
    # Entradas categóricas
    for c in key_cat_cols:
        options = df[c].dropna().unique()
        X_input[c] = st.selectbox(f"{c}", options)
    
    # Crear dataframe con todas las columnas que el modelo espera
    X_pred = pd.DataFrame([{col: 0 if col in num_cols else 'missing' for col in columns_model}])
    
    # Sobrescribir solo las columnas ingresadas
    for col, val in X_input.items():
        X_pred[col] = val
    
    # Transformar categóricas
    for c in cat_cols:
        if c in X_pred.columns:
            X_pred[[c]] = encoder.transform(X_pred[[c]])
    
    # Escalar numéricas
    X_pred[num_cols] = scaler.transform(X_pred[num_cols])
    
    # Predicción
    pred = modelo.predict(X_pred)
    st.success(f"Predicción de digital adoption likelihood: **{pred[0]}**")

# =============================
# Mapeo de páginas
# =============================
pages = {
    "Segmentación Digital": page_segmentacion,
    "Predicción Modelo": page_modelo
}

selected_page = st.sidebar.selectbox("Selecciona la página", list(pages.keys()))
pages[selected_page]()
