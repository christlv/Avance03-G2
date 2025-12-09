import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# ------------------------------
# Configuración global
# ------------------------------
st.set_page_config(page_title="Sesión 2 | ISIL", layout="wide")

# ------------------------------
# Cargar datos (cacheado)
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()

# ------------------------------
# Preprocesamiento (cacheado)
# ------------------------------
@st.cache_data
def preprocess_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    target = "digital_adoption_likelihood"
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    num_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != target]
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target]

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Clip outliers
    cols_to_clip = [
        'TransactionAmount (INR)','CustAccountBalance','DigitalTransactionsCount',
        'BranchTransactionsCount','SpendBalanceRatio','CustomerAge','CustomerTenureYears'
    ]
    for col in cols_to_clip:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(p1, p99)

    # Features adicionales
    def normalize(col):
        if col.max() == col.min():
            return col * 0
        return (col - col.min()) / (col.max() - col.min())

    df["norm_digital_txn"] = normalize(df["DigitalTransactionsCount"])
    df["norm_spend_ratio"] = normalize(df["SpendBalanceRatio"])
    df["norm_tenure"] = normalize(df["CustomerTenureYears"])
    df["DigitalActivityScore"] = df["norm_digital_txn"] + df["norm_spend_ratio"] + df["norm_tenure"]

    # Eliminar columnas de fecha
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    df = df.drop(columns=date_cols)

    return df, num_cols, cat_cols, target

df, num_cols, cat_cols, target = preprocess_data(df)

# ============================================================
# Función para entrenar modelo (cacheado)
# ============================================================
@st.cache_data(show_spinner=True)
def train_model(df, num_cols, cat_cols, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Convertir categóricos a string
    for c in cat_cols:
        X[c] = X[c].astype(str)

    # Encoding
    enc = OrdinalEncoder()
    X[cat_cols] = enc.fit_transform(X[cat_cols])

    # Escalado numérico
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Modelo LightGBM
    modelo = LGBMClassifier(n_estimators=100, learning_rate=0.05)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return modelo, X_test, y_test, report, cm

# ============================================================
# Función para página de modelo
# ============================================================
def page_modelo():
    st.title("Modelo Predictivo: LightGBM")
    st.write("Entrenamiento y evaluación del modelo LightGBM sobre el dataset")

    with st.spinner("Entrenando modelo..."):
        modelo, X_test, y_test, report, cm = train_model(df, num_cols, cat_cols, target)

    st.subheader("📊 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🔹 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)

# ============================================================
# Páginas existentes
# ============================================================
def page_segmentacion():
    st.title("Segmentación de Clientes por Comportamiento Digital | Timeline")
    st.write("EDA - segmentación y análisis del comportamiento digital")
    
    opcion = st.slider("Selecciona un punto del timeline", 1, 5, 1)

    if opcion == 1:
        st.info("Distribución de adopción digital")
        counts = df['digital_adoption_likelihood'].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        counts.plot(kind='bar', color=['skyblue', 'orange'], ax=ax)
        for i, val in enumerate(counts):
            ax.text(i, val + 2, str(val), ha='center', va='bottom')
        st.pyplot(fig, clear_figure=True)

# ============================================================
# Otra página de ejemplo
# ============================================================
def page_otra():
    st.title("Otra página de ejemplo")
    st.write("Contenido independiente.")

# ============================================================
# Menú lateral y mapeo de páginas
# ============================================================
pages = {
    "Segmentación Digital": page_segmentacion,
    "Modelo Predictivo": page_modelo,
    "Otra Página": page_otra,
}

selected_page = st.sidebar.selectbox("Selecciona la página", list(pages.keys()))
pages[selected_page]()
