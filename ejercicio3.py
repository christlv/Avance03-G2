# ============================
# app.py - Streamlit consolidado
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Configuración global
st.set_page_config(page_title="Sesión 2 | ISIL", layout="centered")

# =====================================
# 1. CARGA DE DATOS
# =====================================
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_digital_adoptionv2.xlsx")
    df = df.replace([np.inf, -np.inf], np.nan)
    target = "digital_adoption_likelihood"
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    return df

df = load_data()

# =====================================
# 2. CARGA DEL MODELO Y TRANSFORMADORES
# =====================================
@st.cache_resource
def load_model():
    modelo = pickle.load(open("modelo.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return modelo, encoder, scaler

modelo, encoder, scaler = load_model()

# =====================================
# 3. DEFINICIÓN DE COLUMNAS
# =====================================
target = "digital_adoption_likelihood"
num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
num_cols.remove(target)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# =====================================
# 4. CREAR NUEVAS FEATURES
# =====================================
def normalize(col):
    if col.max() == col.min():
        return col * 0
    return (col - col.min()) / (col.max() - col.min())

df["norm_digital_txn"] = normalize(df["DigitalTransactionsCount"])
df["norm_spend_ratio"] = normalize(df["SpendBalanceRatio"])
df["norm_tenure"] = normalize(df["CustomerTenureYears"])
df["DigitalActivityScore"] = (
    df["norm_digital_txn"] +
    df["norm_spend_ratio"] +
    df["norm_tenure"]
)

# =====================================
# 5. FUNCIONES DE PÁGINA
# =====================================

# ---------- Página 1: Segmentación ----------
def page_segmentacion():
    st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
    st.write("EDA - Segmentación y análisis del comportamiento digital")

    opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

    if opcion == 1:
        st.info("Distribución de adopción digital")
        counts = df[target].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', color=['skyblue', 'orange'], ax=ax)
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
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width()/2., p.get_height()),
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

# ---------- Página 2: Predicción ----------
def page_prediccion():
    st.title("Predicción de Digital Adoption")
    st.write("Modelo entrenado: LightGBM")

    if st.button("Generar predicciones"):
        X_pred = df.drop(columns=[target]).copy()
        # Aplicar transformaciones
        for c in cat_cols:
            X_pred[c] = X_pred[c].astype(str)
        X_pred[cat_cols] = encoder.transform(X_pred[cat_cols])
        X_pred[num_cols] = scaler.transform(X_pred[num_cols])

        pred = modelo.predict(X_pred)
        df_result = df.copy()
        df_result['Predicted_Adoption'] = pred

        st.dataframe(df_result.head(50))

        output_file = "predicciones_digital_adoption.xlsx"
        df_result.to_excel(output_file, index=False)
        st.download_button(
            label="Descargar predicciones",
            data=open(output_file, "rb").read(),
            file_name="predicciones_digital_adoption.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =====================================
# 6. MENÚ LATERAL
# =====================================
pages = {
    "Segmentación Digital": page_segmentacion,
    "Predicción Digital Adoption": page_prediccion
}

selected_page = st.sidebar.selectbox("Selecciona la página", list(pages.keys()))
pages[selected_page]()
