import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend seguro
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="Segmentación Clientes | Timeline", layout="centered")

st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
st.write("Autor: Christian Torres | ISIL")

# ------------------------- CARGA DE DATOS -------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()

# ------------------------- SLIDER -------------------------
opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

# ------------------------- FUNCIÓN AUXILIAR -------------------------
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)  # importante para evitar errores de DOM

# ------------------------- 1️⃣ Distribución adopción digital -------------------------
if opcion == 1:
    st.info("Distribución de adopción digital")
    counts = df['digital_adoption_likelihood'].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)
    ax.set_title("Distribución de adopción digital")
    for i, val in enumerate(counts):
        ax.text(i, val + 2, str(val), ha='center', va='bottom')
    show_fig(fig)

# ------------------------- 2️⃣ Distribución género -------------------------
elif opcion == 2:
    st.info("Distribución de género de clientes")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='CustGender', palette=['pink','skyblue'], ax=ax)
    ax.set_title("Distribución de género de clientes")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom')
    show_fig(fig)

# ------------------------- 3️⃣ Distribución DigitalActivityScore -------------------------
elif opcion == 3:
    st.info("Distribución de Digital Activity Score")
    fig, ax = plt.subplots()
    sns.histplot(df["DigitalActivityScore"], binwidth=0.25, kde=False, ax=ax)  # varias barritas
    ax.set_title("Distribución de Digital Activity Score")
    show_fig(fig)

# ------------------------- 4️⃣ Relación transacciones -------------------------
elif opcion == 4:
    st.info("Relación entre transacciones digitales y presenciales")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x='DigitalTransactionsCount', y='BranchTransactionsCount', ax=ax)
    ax.set_title("Relación entre transacciones digitales y presenciales")
    show_fig(fig)

# ------------------------- 5️⃣ Tipos de tarjeta -------------------------
elif opcion == 5:
    st.info("Tipos de tarjeta por cliente")
    color_map = {'Black':'#000000','Platinum':'#E5E4E2','Gold':'#FFD700','Classic':'#1E90FF'}
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='CreditCardType', palette=color_map, ax=ax)
    ax.set_title("Tipos de tarjeta por cliente")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height + 3),
                    ha='center', va='bottom', fontsize=10,
                    color='white' if p.get_facecolor() == (0,0,0,1) else 'black')
    show_fig(fig)
