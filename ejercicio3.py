import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend seguro para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- CONFIGURACIÓN -------------------------
st.set_page_config(page_title="Sesión 2 | ISIL", layout="centered")

st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
st.write("Autor: Christian Torres | ISIL")
st.write("EDA - segmentación y el análisis del comportamiento digital")

# ------------------------- CARGA DE DATOS -------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()   # ← AHORA SÍ EXISTE

# ------------------------- TIMELINE -------------------------
opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

# Función para asegurar que cada figura esté limpia
def new_fig():
    plt.clf()
    plt.close('all')
    return plt.subplots()

# ------------------------- 1️⃣ DISTRIBUCIÓN ADOPCIÓN DIGITAL -------------------------
if opcion == 1:
    st.info("**Distribución de adopción digital**")

    counts = df['digital_adoption_likelihood'].value_counts()

    fig, ax = new_fig()
    counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)

    ax.set_title("Distribución de adopción digital")

    for i, val in enumerate(counts):
        ax.text(i, val + 2, str(val), ha='center', va='bottom')

    st.pyplot(fig)

    st.code("""
counts = df['digital_adoption_likelihood'].value_counts()
counts.plot(kind='bar', color=['skyblue','orange'])
""", language="python")

# ------------------------- 2️⃣ DISTRIBUCIÓN DE GÉNERO -------------------------
elif opcion == 2:
    st.info("**Distribución de género de clientes**")

    fig, ax = new_fig()
    sns.countplot(data=df, x='CustGender', palette=['pink','skyblue'], ax=ax)
    ax.set_title("Distribución de género de clientes")

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')

    st.pyplot(fig)

    st.code("""
sns.countplot(data=df, x='CustGender')
""", language="python")

# ------------------------- 3️⃣ DISTRIBUCIÓN ACTIVITY SCORE -------------------------
elif opcion == 3:
    st.info("**Distribución de Digital Activity Score**")

    fig, ax = plt.subplots()

    # Usar binwidth fijo para que se vean varias barras
    sns.histplot(
        df["DigitalActivityScore"],
        binwidth=0.25,   # ancho de cada barra
        kde=False,
        ax=ax
    )
    ax.set_title("Distribución de Digital Activity Score")
    st.pyplot(fig)

    st.code("""
sns.histplot(df["DigitalActivityScore"], binwidth=0.25)
plt.title("Distribución de Digital Activity Score")
""", language="python")











# ------------------------- 4️⃣ RELACIÓN DE TRANSACCIONES -------------------------
elif opcion == 4:
    st.info("**Relación entre transacciones digitales y presenciales**")

    fig, ax = new_fig()
    sns.scatterplot(
        data=df,
        x='DigitalTransactionsCount',
        y='BranchTransactionsCount',
        ax=ax
    )
    ax.set_title("Relación entre transacciones digitales y presenciales")

    st.pyplot(fig)

    st.code("""
sns.scatterplot(data=df,
                x='DigitalTransactionsCount',
                y='BranchTransactionsCount')
""", language="python")

# ------------------------- 5️⃣ TIPOS DE TARJETA -------------------------
elif opcion == 5:
    st.info("**Tipos de tarjeta por cliente**")

    color_map = {
        'Black': '#000000',
        'Platinum': '#E5E4E2',
        'Gold': '#FFD700',
        'Classic': '#1E90FF'
    }

    fig, ax = new_fig()
    sns.countplot(data=df, x='CreditCardType', palette=color_map, ax=ax)
    ax.set_title("Tipos de tarjeta por cliente")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width()/2., height + 3),
            ha='center',
            va='bottom',
            fontsize=10,
            color='white' if p.get_facecolor() == (0, 0, 0, 1) else 'black'
        )

    st.pyplot(fig)

    st.code("""
sns.countplot(data=df, x='CreditCardType')
""", language="python")
