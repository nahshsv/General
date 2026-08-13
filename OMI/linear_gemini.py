import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.lines import Line2D

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("csv/africa_gdp_no2_merged_k=0.8.csv")

# =========================
# 2. CLEAN
# =========================
df = df.dropna(subset=["Year", "ISO", "National_Urban_NO2_Sum", "log_GDP"])
df = df[df["National_Urban_NO2_Sum"] > 0]

df["log_NO2"] = np.log(df["National_Urban_NO2_Sum"])

# =========================
# 3. INCOME MAP
# =========================
income_map = {
    "SYC":"High income",
    "BFA":"Low income","BDI":"Low income","CAF":"Low income","TCD":"Low income",
    "COD":"Low income","ERI":"Low income","GMB":"Low income","GNB":"Low income",
    "LBR":"Low income","MDG":"Low income","MWI":"Low income","MLI":"Low income",
    "MOZ":"Low income","NER":"Low income","RWA":"Low income","SLE":"Low income",
    "SOM":"Low income","SDS":"Low income","SDN":"Low income","TGO":"Low income",
    "UGA":"Low income",
    "AGO":"Lower middle income","BEN":"Lower middle income","CMR":"Lower middle income",
    "COM":"Lower middle income","COG":"Lower middle income","CIV":"Lower middle income",
    "SWZ":"Lower middle income","GHA":"Lower middle income","GIN":"Lower middle income",
    "KEN":"Lower middle income","LSO":"Lower middle income","MRT":"Lower middle income",
    "NAM":"Lower middle income","NGA":"Lower middle income","STP":"Lower middle income",
    "SEN":"Lower middle income","TZA":"Lower middle income","ZMB":"Lower middle income",
    "ZWE":"Lower middle income",
    "BWA":"Upper middle income","CPV":"Upper middle income","GNQ":"Upper middle income",
    "GAB":"Upper middle income","MUS":"Upper middle income","ZAF":"Upper middle income",
    "ETH":"Upper middle income"
}

df["income_group"] = df["ISO"].map(income_map)

# =========================
# 4. COLORS
# =========================
colors = {
    "Low income": "#1f77b4",
    "Lower middle income": "#ff7f0e",
    "Upper middle income": "#2ca02c",
    "High income": "#d62728"
}

# =========================
# 5. GLOBAL REGRESSION
# =========================
X = sm.add_constant(df["log_NO2"])
y = df["log_GDP"]

model = sm.OLS(y, X).fit()

beta0 = model.params["const"]
beta1 = model.params["log_NO2"]
r2 = model.rsquared
pval = model.pvalues["log_NO2"]
n = int(model.nobs)

print("\n===== GLOBAL MODEL =====")
print(f"n = {n}")
print(f"β1 = {beta1:.4f}")
print(f"R² = {r2:.4f}")
print(f"p-value = {pval:.4e}")

# =========================
# 6. PLOT
# =========================
plt.figure(figsize=(11,7))

# --- (A) TRACE POINTS ---
for iso, group in df.groupby("ISO"):
    g = group.sort_values("Year")
    income = g["income_group"].iloc[0]

    alphas = np.linspace(0.1, 0.7, len(g))
    sizes = np.linspace(10, 35, len(g))

    for i in range(len(g)):
        plt.scatter(
            g["log_NO2"].iloc[i],
            g["log_GDP"].iloc[i],
            color=colors.get(income, "gray"),
            alpha=alphas[i],
            s=sizes[i]
        )

# --- (B) REGRESSION BY INCOME GROUP ---
print("\n===== BY INCOME GROUP =====")

for income, color in colors.items():
    subset = df[df["income_group"] == income]

    if len(subset) < 10:
        continue

    X_g = sm.add_constant(subset["log_NO2"])
    y_g = subset["log_GDP"]

    model_g = sm.OLS(y_g, X_g).fit()

    b0 = model_g.params["const"]
    b1 = model_g.params["log_NO2"]
    r2_g = model_g.rsquared

    print(f"{income}: β={b1:.3f}, R²={r2_g:.3f}")

    x_vals = np.linspace(subset["log_NO2"].min(), subset["log_NO2"].max(), 200)
    y_vals = b0 + b1 * x_vals

    plt.plot(
        x_vals,
        y_vals,
        color=color,
        linewidth=3,
        label=f"{income}"
    )

# --- (C) GLOBAL LINE (optional dashed) ---
x_vals_global = np.linspace(df["log_NO2"].min(), df["log_NO2"].max(), 200)
y_vals_global = beta0 + beta1 * x_vals_global

plt.plot(
    x_vals_global,
    y_vals_global,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Global"
)

# --- (D) STATS BOX ---
textstr = f'Global:\nβ₁ = {beta1:.2f}\nR² = {r2:.2f}\np = {pval:.2e}\nn = {n}'

plt.text(
    0.02, 0.98, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)

# --- (E) LEGEND CLEAN ---
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), title="Income Group")

# =========================
# 7. STYLE
# =========================
plt.xlabel("log(NO₂ concentration)", fontsize=12)
plt.ylabel("log(GDP)", fontsize=12)
plt.title("NO₂–GDP Relationship by Income Group in Africa", fontsize=14)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("final_plot_income_regression.png", dpi=300)
plt.show()