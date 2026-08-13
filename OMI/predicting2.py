import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("csv/africa_gdp_no2_merged_k=0.8.csv")

# =========================
# 2. CLEAN
# =========================
df = df.dropna(subset=["Year", "ISO", "National_Urban_NO2_Sum"])
df = df[df["National_Urban_NO2_Sum"] > 0]

df["log_NO2"] = np.log(df["National_Urban_NO2_Sum"])

# =========================
# 3. SPLIT DATA
# =========================
train = df[(df["Year"] >= 2005) & (df["Year"] <= 2022)].copy()
test  = df[(df["Year"] >= 2023) & (df["Year"] <= 2024)].copy()

# cần GDP cho train/test
train = train.dropna(subset=["log_GDP"])
test  = test.dropna(subset=["log_GDP"])

# =========================
# 4. TRAIN MODEL
# =========================
X_train = sm.add_constant(train["log_NO2"])
y_train = train["log_GDP"]

model = sm.OLS(y_train, X_train).fit()

beta0 = model.params["const"]
beta1 = model.params["log_NO2"]

print("\n===== MODEL =====")
print(f"β0 = {beta0:.4f}")
print(f"β1 = {beta1:.4f}")
print(f"R² = {model.rsquared:.4f}")

# =========================
# 5. CALCULATE BIAS (OUT-OF-SAMPLE)
# =========================
test["log_GDP_pred"] = beta0 + beta1 * test["log_NO2"]
test["bias"] = test["log_GDP"] - test["log_GDP_pred"]

bias_mean = test["bias"].mean()
bias_median = test["bias"].median()

print("\n===== OUT-OF-SAMPLE BIAS =====")
print(f"Bias (mean)   = {bias_mean:.4f}")
print(f"Bias (median) = {bias_median:.4f}")

# chọn bias
bias = bias_mean

# =========================
# 6. PREDICT 2025
# =========================
df_2025 = df[df["Year"] == 2025].copy()

df_2025 = df_2025.dropna(subset=["National_Urban_NO2_Sum"])
df_2025 = df_2025[df_2025["National_Urban_NO2_Sum"] > 0]

df_2025["log_NO2"] = np.log(df_2025["National_Urban_NO2_Sum"])

# raw prediction
df_2025["log_GDP_pred"] = beta0 + beta1 * df_2025["log_NO2"]
df_2025["GDP_pred"] = np.exp(df_2025["log_GDP_pred"])

# bias corrected
df_2025["log_GDP_pred_adj"] = df_2025["log_GDP_pred"] + bias
df_2025["GDP_pred_adj"] = np.exp(df_2025["log_GDP_pred_adj"])

# =========================
# 7. OUTPUT
# =========================
result = df_2025[[
    "ISO",
    "Country_x",
    "National_Urban_NO2_Sum",
    "GDP_pred",
    "GDP_pred_adj"
]]

print("\n===== SAMPLE OUTPUT =====")
print(result.head())

result.to_csv("gdp_2025_predictions_bias_corrected.csv", index=False)

print("\n✅ Saved: gdp_2025_predictions_bias_corrected.csv")