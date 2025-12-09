import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==============================
# Configuración de la página
# ==============================
st.set_page_config(page_title="Sesión 2 | ISIL", layout="centered")

# ==============================
# Funciones de carga
# ==============================
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

@st.cache_data
def load_model_files():
    try:
        files = ["modelo_lightgbm.pkl", "encoder.pkl", "scaler.pkl", "num_cols.pkl", "cat_cols.pkl"]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            st.error(f"No se encontraron los archivos: {', '.join(missing)}")
            return None, None, None, None, None
        
        modelo = joblib.load("modelo_lightgbm.pkl")
        encoder = joblib.load("encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        num_cols = joblib.load("num_cols.pkl")
        cat_cols = joblib.load("cat_cols.pkl")
        st.success("✅ Modelo y transformadores cargados correctamente")
        return modelo, encoder, scaler, num_cols, cat_cols
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None, None, None, None

# ==============================
# Cargar datos y modelos
# ==============================
df = load_data()
modelo, encoder, scaler, num_cols, cat_cols = load_model_files()
target = "digital_adoption_likelihood"

# ==============================
# Preprocesamiento inicial para gráficas
# ==============================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

num_cols_all = [c for c in df.select_dtypes(include=['float64','int64']).columns if c != target]
cat_cols_all = [c for c in df.select_dtypes(include=['object']).columns if c != target]

df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median())
for c in cat_cols_all:
    df[c] = df[c].fillna(df[c].mode()[0])

# Clip de outliers
cols_to_clip = [
    'TransactionAmount (INR)',
    'CustAccountBalance',
    'DigitalTransactionsCount',
    'BranchTransactionsCount',
    'SpendBalanceRatio',
    'CustomerAge',
    'CustomerTenureYears'
]
for col in cols_to_clip:
    p1 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

# Features adicionales
def normalize(col):
    if col.max() == col.min():
        return col * 0
    return (col - col.min()) / (col.max() - col.min())

df["norm_digital_txn"] = normalize(df["DigitalTransactionsCount"])
df["norm_spend_ratio"] = normalize(df["SpendBalanceRatio"])
df["norm_tenure"] = normalize(df["CustomerTenureYears"])
df["DigitalActivityScore"] = df["norm_digital_txn"] + df["norm_spend_ratio"] + df["norm_tenure"]

# ==============================
# Página de gráficas
# ==============================
def page_segmentacion():
    st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
    st.write("Autor: Christian Torres | ISIL")
    st.write("EDA - segmentación y análisis del comportamiento digital")

    opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

    if opcion == 1:
        st.info("Distribución de adopción digital")
        counts = df[target].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)
        ax.set_title("Distribución de adopción digital")
        for i, val in enumerate(counts):
            ax.text(i, val + 2, str(val), ha='center', va='bottom')
        st.pyplot(fig)

    elif opcion == 2:
        st.info("Distribución de género de clientes")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='CustGender', palette=['pink','skyblue'], ax=ax)
        ax.set_title("Distribución de género")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom')
        st.pyplot(fig)

    elif opcion == 3:
        st.info("Distribución de Digital Activity Score")
        data = df['DigitalActivityScore'].dropna()
        data = data[(data >= 0) & (data <= 3)]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data, bins=30, kde=False, ax=ax)
        ax.set_title("Distribución de Digital Activity Score")
        ax.set_xlabel("DigitalActivityScore")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 14000)
        ax.set_yticks(range(0, 15000, 2000))
        st.pyplot(fig)

    elif opcion == 4:
        st.info("Relación entre transacciones digitales y presenciales")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=df, x='DigitalTransactionsCount', y='BranchTransactionsCount', ax=ax)
        ax.set_title("Transacciones digitales vs presenciales")
        st.pyplot(fig)

    elif opcion == 5:
        st.info("Tipos de tarjeta por cliente")
        color_map = {'Black':'#000000', 'Platinum':'#E5E4E2', 'Gold':'#FFD700', 'Classic':'#1E90FF'}
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='CreditCardType', palette=color_map, ax=ax)
        ax.set_title("Tipos de tarjeta por cliente")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height + 3),
                        ha='center', va='bottom',
                        fontsize=10,
                        color='white' if p.get_facecolor() == (0,0,0,1) else 'black')
        st.pyplot(fig)

# ==============================
# Página de predicción
# ==============================
def page_modelo():
    st.title("Predicción de Adopción Digital")
    st.write("Ingrese los valores de las características del cliente:")

    if modelo is None:
        st.warning("Modelo no cargado. No se puede realizar predicción.")
        return

    # Crear un diccionario de inputs
    input_data = {}
    for col in num_cols:
        val = st.number_input(col, value=0)
        input_data[col] = val
    for col in cat_cols:
        val = st.text_input(col, value="")
        input_data[col] = val

    X_pred = pd.DataFrame([input_data])

    # 🔹 Transformar categóricas con el encoder (todas a la vez)
    X_pred[cat_cols] = encoder.transform(X_pred[cat_cols])

    # 🔹 Escalar numéricas
    X_pred[num_cols] = scaler.transform(X_pred[num_cols])

    # Predicción
    pred = modelo.predict(X_pred)
    st.write(f"Predicción de digital adoption likelihood: **{pred[0]}**")

# ==============================
# Mapeo de páginas
# ==============================
pages = {
    "Segmentación Digital": page_segmentacion,
    "Predicción Modelo": page_modelo,
}

selected_page = st.sidebar.selectbox("Selecciona la página", list(pages.keys()))
pages[selected_page]()
