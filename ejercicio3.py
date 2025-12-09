import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración global
st.set_page_config(page_title="Sesión 2 | ISIL", layout="centered")

# Cargar datos (cacheado)
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()

# ============================================================
#BLOQUE NUEVO DE PREPROCESAMIENTO
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

# ------------------------------
# 1. Definir variable objetivo
# ------------------------------
target = "digital_adoption_likelihood"

# Eliminar filas donde el target es NA
df = df.dropna(subset=[target])

# Asegurar tipo
df[target] = df[target].astype(int)

# ------------------------------
# 2. Imputación correcta
# ------------------------------
num_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != target]
cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target]

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------
# 3. Outliers
# ------------------------------
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

# ------------------------------
# 4. Features adicionales
# ------------------------------
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

# ============================================================
# 🔥 FIN DEL BLOQUE AGREGADO
# ============================================================


# Función para mostrar gráficos de la página 1 (tu gráfico actual con sliders)
def page_segmentacion():
    st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
    st.write("Autor: Christian Torres | ISIL")
    st.write("EDA - segmentación y análisis del comportamiento digital")
    
    opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)
    
    if opcion == 1:
        st.info("Distribución de adopción digital")
        counts = df['digital_adoption_likelihood'].value_counts()
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

# Función ejemplo para otra página
def page_otra():
    st.title("Otra página de ejemplo")
    st.write("Aquí podrías poner otro análisis o contenido independiente.")
    # Aquí podrías poner otro slider o controles propios

# Mapeo de páginas con su función
pages = {
    "Segmentación Digital": page_segmentacion,
    "Otra Página": page_otra,
    # Puedes agregar más funciones para más páginas
}

# Menú lateral para selección de página
selected_page = st.sidebar.selectbox("Selecciona la página", list(pages.keys()))

# Ejecutar la función correspondiente a la página seleccionada
pages[selected_page]()
