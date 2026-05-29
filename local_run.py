"""
Local runner: preprocesa `Viviendas_limpio.csv`, ejecuta KMeans (sklearn)
y guarda `viviendas_local_with_clusters.csv`.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

CSV = "Viviendas_limpio.csv"
OUT = "viviendas_local_with_clusters.csv"
N_CLUSTERS = 4

if not os.path.exists(CSV):
    print(f"ERROR: no se encontró {CSV} en el directorio actual.")
    sys.exit(2)

print(f"Cargando {CSV}...")
df = pd.read_csv(CSV)
print(f"Registros: {df.shape[0]}, columnas: {df.shape[1]}")

# Limpiar columnas esperadas
df = df.drop(columns=["Unnamed: 0", "Transaccion"], errors="ignore")

# Encode variables categóricas
le = LabelEncoder()
for col in ["Calefaccion", "Ascensor", "Garaje", "VPO"]:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Imputar Antiguedad
if "Antiguedad" in df.columns:
    df["Antiguedad"] = df["Antiguedad"].fillna(df["Antiguedad"].median())

feature_cols = [c for c in ["Precio", "Superficie", "Antiguedad", "VPO",
                             "Calefaccion", "Habitaciones", "Ascensor",
                             "Garaje", "Precio_m2"] if c in df.columns]
if len(feature_cols) == 0:
    print("ERROR: no se encontraron columnas de features esperadas.")
    sys.exit(3)

# Limpiar y convertir a numérico columnas de features (manejar textos como 'Entre 2 y 3')
for col in feature_cols:
    # Reemplazar comas decimales y extraer la primera ocurrencia numérica
    cleaned = df[col].astype(str).str.replace(',', '.').str.extract(r'([0-9]+\.?[0-9]*)')[0]
    df[col] = pd.to_numeric(cleaned, errors='coerce')
    # Si quedan NaN, imputar con la mediana (o 0 si la mediana no está definida)
    if df[col].isna().all():
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(df[col].median())

X = df[feature_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Entrenando KMeans (sklearn) localmente...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels

# Guardar resultado
df.to_csv(OUT, index=False)
print(f"✅ Resultado guardado en {OUT}")
print("Distribución de clusters:")
print(pd.Series(labels).value_counts().sort_index())
print("Hecho.")
