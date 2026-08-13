import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
# 5. GLOBAL REGRESSION (OLS)
# =========================
X = sm.add_constant(df["log_NO2"])
y = df["log_GDP"]

model = sm.OLS(y, X).fit()

beta0 = model.params["const"]
beta1 = model.params["log_NO2"]
r2 = model.rsquared
pval = model.pvalues["log_NO2"]
n = int(model.nobs)

print("\n===== SUMMARY =====")
print(f"n = {n}")
print(f"β1 = {beta1:.4f}")
print(f"R² = {r2:.4f}")
print(f"p-value = {pval:.4e}")

# regression line
x_vals = np.linspace(df["log_NO2"].min(), df["log_NO2"].max(), 100)
y_vals = beta0 + beta1 * x_vals

# =========================
# 6. PLOT
# =========================
plt.figure(figsize=(10,7))

# --- (A) TRACE (fade dots, KHÔNG nối) ---
for iso, group in df.groupby("ISO"):
    g = group.sort_values("Year")
    income = g["income_group"].iloc[0]
    
    alphas = np.linspace(0.1, 0.7, len(g))
    sizes = np.linspace(10, 35, len(g))  # optional đẹp hơn
    
    for i in range(len(g)):
        plt.scatter(
            g["log_NO2"].iloc[i],
            g["log_GDP"].iloc[i],
            color=colors.get(income, "gray"),
            alpha=alphas[i],
            s=sizes[i]
        )

# --- (B) SCATTER (legend sạch, tránh lặp label) ---
for income, color in colors.items():
    subset = df[df["income_group"] == income]
    if len(subset) > 0:
        plt.scatter(
            subset["log_NO2"],
            subset["log_GDP"],
            color=color,
            alpha=0.4,
            label=income
        )

# --- (C) GLOBAL REGRESSION LINE ---
plt.plot(
    x_vals,
    y_vals,
    color="black",
    linewidth=3,
)

# --- (D) STATS BOX ---
print(f"β1 = {beta1:.2f}, R² = {r2:.2f}, p = {pval:.2e}, n = {n}")

# =========================
# 7. STYLE
# =========================
plt.xlabel("log(NO₂)")
plt.ylabel("log(GDP)")
plt.title("NO₂–GDP Relationship in Africa", fontsize=14)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("final_plot.png", dpi=300)
plt.show()