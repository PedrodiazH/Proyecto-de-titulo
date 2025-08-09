import matplotlib.pyplot as plt

# Datos
years = [2024, 2029]
market_size = [8.58, 18.28]  # en miles de millones USD
cagr = 16.20  # porcentaje

# Configuración estilo académico
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# Crear figura
fig, ax = plt.subplots(figsize=(6, 4))

# Posiciones de barras
bar_positions = [0, 1]
bars = ax.bar(bar_positions, market_size, width=0.5,
              color="#7BAFD4", edgecolor="black")

# Eje X
ax.set_xticks(bar_positions)
ax.set_xticklabels(years)
ax.set_ylabel("Tamaño del mercado (USD Billones)")
#ax.set_title("Facial Recognition Market Size (USD Billion)", weight='bold')
ax.set_ylim(0, max(market_size) + 5)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Valores sobre las barras
for bar, value in zip(bars, market_size):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.3,
            f"{value:.2f}", ha='center', va='bottom', fontsize=11)

# Línea segmentada horizontal desde barra 2024 hasta barra 2029
ax.plot([bar_positions[0] + 0.25, bar_positions[1] - 0.25],
        [market_size[0], market_size[0]],
        linestyle="--", color="black")

# Texto en esquina superior izquierda
ax.text(0.02, 0.96, f"CAGR {cagr:.2f}%",
        fontsize=12, fontweight='bold', ha='left', va='top',
        transform=ax.transAxes)
ax.text(0.02, 0.90, "(Tasa de crecimiento anual compuesta)",
        fontsize=9, style='italic', ha='left', va='top',
        transform=ax.transAxes)

plt.tight_layout()
plt.show()
