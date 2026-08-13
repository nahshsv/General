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
# 3. INCOME MAP (Lower middle only)
# =========================
lm_list = [
    "AGO","BEN","CMR","COM","COG","CIV","SWZ","GHA","GIN",
    "KEN","LSO","MRT","NAM","NGA","STP","SEN","TZA","ZMB","ZWE"
]

df_lm = df[df["ISO"].isin(lm_list)]

# =========================
# 4. REGRESSION
# =========================
X = sm.add_constant(df_lm["log_NO2"])
y = df_lm["log_GDP"]

model = sm.OLS(y, X).fit()
beta0 = model.params["const"]
beta1 = model.params["log_NO2"]

# =========================
# 5. PLOT
# =========================
plt.style.use('default')
plt.figure(figsize=(9,6))

# --- TRACE FADE (KHÔNG LINE) ---
for iso, group in df_lm.groupby("ISO"):
    g = group.sort_values("Year")

    alphas = np.linspace(0.1, 0.9, len(g))   # fade theo thời gian
    sizes = np.linspace(15, 50, len(g))      # optional: lớn dần

    for i in range(len(g)):
        plt.scatter(
            g["log_NO2"].iloc[i],
            g["log_GDP"].iloc[i],
            color="#ff7f0e",
            alpha=alphas[i],
            s=sizes[i],
            edgecolors='none'
        )

# --- REGRESSION LINE ---
x_vals = np.linspace(
    df_lm["log_NO2"].quantile(0.05),
    df_lm["log_NO2"].quantile(0.95),
    200
)

y_vals = beta0 + beta1 * x_vals

plt.plot(
    x_vals,
    y_vals,
    color="#d95f02",
    linewidth=3
)

# =========================
# 6. ZOOM
# =========================
plt.xlim(df_lm["log_NO2"].quantile(0.02), df_lm["log_NO2"].quantile(0.98))
plt.ylim(df_lm["log_GDP"].quantile(0.02), df_lm["log_GDP"].quantile(0.98))

# =========================
# 7. STYLE
# =========================
plt.xlabel("log(NO₂)", fontsize=11)
plt.ylabel("log(GDP)", fontsize=11)

plt.title(
    "NO₂–GDP Relationship in Lower-Middle-Income Countries",
    fontsize=13,
    pad=10
)

plt.grid(alpha=0.15)
plt.tight_layout()

# =========================
# 8. SAVE
# =========================
plt.savefig("lower_middle_trace_fade.png", dpi=300, bbox_inches='tight')

plt.show()
beta0 = model.params["const"]
beta1 = model.params["log_NO2"]

print("\n===== REGRESSION COEFFICIENTS =====")
print(f"β₀ (intercept) = {beta0:.4f}")
print(f"β₁ (slope)     = {beta1:.4f}")