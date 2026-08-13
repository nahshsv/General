# =====================================
# GLOBAL TREND ANALYSIS: NO2 & GDP
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =====================================
# 1. LOAD DATA
# =====================================
file_path = "csv/africa_gdp_no2_merged_k=0.8.csv"   # <-- đổi path tại đây
df = pd.read_csv(file_path)

# =====================================
# 2. CLEAN DATA
# =====================================
df = df.dropna(subset=["National_Urban_NO2_Sum", "log_GDP"])

df = df.rename(columns={
    "National_Urban_NO2_Sum": "NO2"
})

# Log transform (IMPORTANT)
df["NO2_log"] = np.log(df["NO2"])

# =====================================
# 3. AGGREGATE THEO NĂM (GLOBAL AFRICA)
# =====================================
df_year = df.groupby("Year").agg({
    "NO2_log": "mean",
    "log_GDP": "mean"
}).reset_index()

print("\n=== Aggregated Data ===")
print(df_year.head())

# =====================================
# 4. PLOT TREND
# =====================================
plt.figure()

plt.plot(df_year["Year"], df_year["NO2_log"], marker='o', label="NO2 (log)")
plt.plot(df_year["Year"], df_year["log_GDP"], marker='o', label="GDP_log")

plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Global Trend of NO2 and GDP (Africa)")
plt.legend()

plt.show()

# =====================================
# 5. REGRESSION TEST (TREND CHECK)
# =====================================

# Add constant
X = sm.add_constant(df_year["Year"])

# -------- NO2 TREND --------
y_no2 = df_year["NO2_log"]
model_no2 = sm.OLS(y_no2, X).fit()

print("\n=== NO2 TREND RESULT ===")
print(model_no2.summary())

# -------- GDP TREND --------
y_gdp = df_year["log_GDP"]
model_gdp = sm.OLS(y_gdp, X).fit()

print("\n=== GDP TREND RESULT ===")
print(model_gdp.summary())

# =====================================
# 6. VISUALIZE WITH TREND LINE
# =====================================
plt.figure()

# Scatter
plt.scatter(df_year["Year"], df_year["NO2_log"], label="NO2 (log)")
plt.scatter(df_year["Year"], df_year["log_GDP"], label="GDP_log")

# Regression lines
plt.plot(df_year["Year"], model_no2.predict(X), linestyle='--', label="NO2 trend")
plt.plot(df_year["Year"], model_gdp.predict(X), linestyle='--', label="GDP trend")

plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Trend with Regression Lines")
plt.legend()

plt.show()

# =====================================
# 7. SIMPLE INTERPRETATION (AUTO PRINT)
# =====================================

def interpret(model, name):
    coef = model.params["Year"]
    pval = model.pvalues["Year"]

    print(f"\n--- {name} ---")
    print(f"Slope (Year): {coef:.4f}")
    print(f"P-value: {pval:.5f}")

    if pval < 0.05:
        if coef > 0:
            print(f"→ {name} is SIGNIFICANTLY INCREASING over time")
        else:
            print(f"→ {name} is SIGNIFICANTLY DECREASING over time")
    else:
        print(f"→ No significant trend detected")

interpret(model_no2, "NO2")
interpret(model_gdp, "GDP")