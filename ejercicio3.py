import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# ======================================
# CONFIGURACIÓN STREAMLIT
# ======================================
st.set_page_config(page_title="ISIL | Digital Adoption", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_excel("dataset_digital_adoptionv2.xlsx")

df = load_data()

# ======================================
# PREPROCESAMIENTO
# ======================================
df = df.replace([np.inf, -np.inf], np.nan)

target = "digital_adoption_likelihood"
df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols.remove(target)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

clip_cols = [
    'TransactionAmount (INR)','CustAccountBalance','DigitalTransactionsCount',
    'BranchTransactionsCount','SpendBalanceRatio','CustomerAge','CustomerTenureYears'
]
for c in clip_cols:
    p1, p99 = df[c].quantile([0.01, 0.99])
    df[c] = df[c].clip(p1, p99)

# Nuevas features
def normalize(col):
    if col.max() == col.min():
        return col * 0
    return (col - col.min()) / (col.max() - col.min())

df["norm_digital_txn"] = normalize(df["DigitalTransactionsCount"])
df["norm_spend_ratio"] = normalize(df["SpendBalanceRatio"])
df["norm_tenure"] = normalize(df["CustomerTenureYears"])
df["DigitalActivityScore"] = df["norm_digital_txn"] + df["norm_spend_ratio"] + df["norm_tenure"]

# Eliminar columnas tipo fecha
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
df = df.drop(columns=date_cols)

# ======================================
# FUNCIÓN PARA ENTRENAR MODELO
# ======================================
def entrenar_modelo(df, modelo_nombre="LightGBM"):
    X = df.drop(columns=[target])
    y = df[target]
    
    # Convertir categóricas a string
    for c in cat_cols:
        X[c] = X[c].astype(str)
    
    # Encoding + escalado
    enc = OrdinalEncoder()
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    modelos = {
        "LightGBM": lgb.LGBMClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=300)
    }
    
    modelo = modelos[modelo_nombre]
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    
    # GridSearchCV
    param_grids = {
        "LightGBM": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
        "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "LogisticRegression": {"C": [0.5, 1, 5], "solver": ["lbfgs"]}
    }
    
    grid = GridSearchCV(modelo, param_grids[modelo_nombre], cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    pred_best = grid.best_estimator_.predict(X_test)
    
    return classification_report(y_test, pred_best, output_dict=True), confusion_matrix(y_test, pred_best)

# ======================================
# STREAMLIT INTERFAZ
# ======================================
st.title("ISIL | Segmentación Digital y Modelado")

menu = st.sidebar.selectbox("Menú", ["EDA", "Modelo"])

if menu == "EDA":
    st.header("Exploración de Datos")
    opcion = st.slider("Selecciona un gráfico", 1, 5, 1)
    
    if opcion == 1:
        st.subheader("Distribución adopción digital")
        counts = df[target].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', color=['skyblue','orange'], ax=ax)
        st.pyplot(fig)
    
    elif opcion == 2:
        st.subheader("Distribución de género")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='CustGender', palette=['pink','skyblue'], ax=ax)
        st.pyplot(fig)
    
    elif opcion == 3:
        st.subheader("Digital Activity Score")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df["DigitalActivityScore"], bins=30, kde=False, ax=ax)
        st.pyplot(fig)
    
    elif opcion == 4:
        st.subheader("Transacciones digitales vs presenciales")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=df, x='DigitalTransactionsCount', y='BranchTransactionsCount', ax=ax)
        st.pyplot(fig)
    
    elif opcion == 5:
        st.subheader("Tipos de tarjeta")
        fig, ax = plt.subplots()
        color_map = {'Black':'#000000', 'Platinum':'#E5E4E2','Gold':'#FFD700','Classic':'#1E90FF'}
        sns.countplot(data=df, x='CreditCardType', palette=color_map, ax=ax)
        st.pyplot(fig)

elif menu == "Modelo":
    st.header("Entrenamiento de modelo")
    modelo_nombre = st.selectbox("Selecciona modelo", ["LightGBM", "RandomForest", "GradientBoosting", "LogisticRegression"])
    
    st.write(f"Entrenando {modelo_nombre}... Esto puede tardar unos segundos.")
    
    report, cm = entrenar_modelo(df, modelo_nombre)
    
    st.subheader("Métricas")
    st.json(report)
    
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
