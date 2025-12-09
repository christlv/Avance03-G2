import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sesión 2 | ISIL", layout="centered")

st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
st.write("Autor: Christian Torres | ISIL")
st.write("EDA - segmentación y el análisis del comportamiento digital")

# Cargar datos (ajusta la ruta si es necesario)
# df = pd.read_csv("ruta/dataset.csv")

opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

# --------------------------------------------------------------------
# 1️⃣ DISTRIBUCIÓN DE ADOPCIÓN DIGITAL
# --------------------------------------------------------------------
if opcion == 1:
    st.info("**Distribución de adopción digital**")

    counts = df['digital_adoption_likelihood'].value_counts()

    fig, ax = plt.subplots()
    counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)

    ax.set_title("Distribución de adopción digital")

    # Etiquetas
    for i, val in enumerate(counts):
        ax.text(i, val + 5, str(val), ha='center', va='bottom')

    st.pyplot(fig)

    st.code("""
counts = df['digital_adoption_likelihood'].value_counts()
counts.plot(kind='bar', color=['skyblue','orange'])
""", language="python")

# --------------------------------------------------------------------
# 2️⃣ DISTRIBUCIÓN DE GÉNERO
# --------------------------------------------------------------------
elif opcion == 2:
    st.info("**Distribución de género de clientes**")

    fig, ax = plt.subplots()
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

# --------------------------------------------------------------------
# 3️⃣ DISTRIBUCIÓN DIGITAL ACTIVITY SCORE
# --------------------------------------------------------------------
elif opcion == 3:
    st.info("**Distribución de Digital Activity Score**")

    fig, ax = plt.subplots()
    sns.histplot(df['DigitalActivityScore'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribución de Digital Activity Score")

    st.pyplot(fig)

    st.code("""
sns.histplot(df['DigitalActivityScore'], bins=30)
""", language="python")

# --------------------------------------------------------------------
# 4️⃣ RELACIÓN ENTRE TRANSACCIONES
# --------------------------------------------------------------------
elif opcion == 4:
    st.info("**Relación entre transacciones digitales y presenciales**")

    fig, ax = plt.subplots(figsize=(6,4))
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

# --------------------------------------------------------------------
# 5️⃣ TIPOS DE TARJETA
# --------------------------------------------------------------------
elif opcion == 5:
    st.info("**Tipos de tarjeta por cliente**")

    color_map = {
        'Black': '#000000',
        'Platinum': '#E5E4E2',
        'Gold': '#FFD700',
        'Classic': '#1E90FF'
    }

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='CreditCardType', palette=color_map, ax=ax)
    ax.set_title("Tipos de tarjeta por cliente")

    # Etiquetas
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width()/2., height + 5),
                    ha='center', va='bottom',
                    fontsize=10,
                    color='white' if p.get_facecolor() == (0,0,0,1) else 'black')

    st.pyplot(fig)

    st.code("""
sns.countplot(data=df, x='CreditCardType')
""", language="python")

