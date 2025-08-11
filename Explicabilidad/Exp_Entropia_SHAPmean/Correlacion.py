import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Cargar tu CSV
df = pd.read_csv("Explicabilidad/Exp_Entropia_SHAPmean/shap_metrica_700.csv")

# Calcular confianza (probabilidad máxima)
probs_cols = [col for col in df.columns if col.startswith("prob_")]
df["confianza"] = df[probs_cols].max(axis=1)

# Calcular entropía de la distribución de probabilidad
df["entropia"] = df[probs_cols].apply(lambda row: entropy(row + 1e-10, base=2), axis=1)

# Calcular SHAP mean (promedio de importancias absolutas)
shap_cols = [col for col in df.columns if col.startswith("shap_")]
df["shap_mean"] = df[shap_cols].mean(axis=1)

# ¿Predijo bien o no?
df["acierto"] = df["clase_esperada"] == df["clase_predicha"]

# Reporte numérico global
print("\n--- MÉTRICAS GLOBALES ---")
print(f"Confianza promedio: {df['confianza'].mean():.4f}")
print(f"Entropía promedio: {df['entropia'].mean():.4f}")
print(f"SHAP mean promedio: {df['shap_mean'].mean():.6f}")
print(f"Aciertos totales: {df['acierto'].sum()} de {len(df)}")
print(f"Accuracy: {df['acierto'].mean()*100:.2f}%")

# Reporte por clase
print("\n--- MÉTRICAS POR CLASE ESPERADA ---")
por_clase = df.groupby("clase_esperada").agg({
    "confianza": "mean",
    "entropia": "mean",
    "shap_mean": "mean",
    "acierto": ["mean", "sum", "count"]
})
print(por_clase)

# Calcular varianza de SHAP por clase esperada
varianza_shap = df.groupby("clase_esperada")[shap_cols].var().mean(axis=1)
entropia_promedio = df.groupby("clase_esperada")["entropia"].mean()

# Crear DataFrame con ambas métricas
df_corr = pd.DataFrame({
    "entropia_promedio": entropia_promedio,
    "varianza_shap": varianza_shap
})
print("\n--- CORRELACIÓN ENTRE VARIANZA SHAP Y ENTROPÍA PROMEDIO POR CLASE ---")
print(df_corr.corr())

# Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(df_corr["entropia_promedio"], df_corr["varianza_shap"], s=80)

# Etiquetas de clase
for clase, row in df_corr.iterrows():
    plt.text(row["entropia_promedio"] + 0.01, row["varianza_shap"], str(clase), fontsize=9)

plt.title("Entropía promedio vs. Varianza SHAP por clase")
plt.xlabel("Entropía promedio")
plt.ylabel("Varianza de SHAP")
plt.grid(True)
plt.tight_layout()
plt.show()

print(df_corr)

