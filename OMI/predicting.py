import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# 1. LOAD TRAINING DATA (<= 2024)
# =========================
df = pd.read_csv("csv/africa_gdp_no2_merged_k=0.8.csv")

# dùng data có GDP để train (loại 2025 nếu chưa có GDP)
df_train = df[df["Year"] < 2025].copy()

# clean
df_train = df_train.dropna(subset=["National_Urban_NO2_Sum", "log_GDP"])
df_train = df_train[df_train["National_Urban_NO2_Sum"] > 0]

# log NO2
df_train["log_NO2"] = np.log(df_train["National_Urban_NO2_Sum"])

# =========================
# 2. TRAIN MODEL (OLS)
# =========================
X = sm.add_constant(df_train["log_NO2"])
y = df_train["log_GDP"]

model = sm.OLS(y, X).fit()

beta0 = model.params["const"]
beta1 = model.params["log_NO2"]

print("\n===== TRAINED MODEL =====")
print(f"β0 = {beta0:.4f}")
print(f"β1 = {beta1:.4f}")
print(f"R² = {model.rsquared:.4f}")

# =========================
# 3. LOAD 2025 DATA (ONLY NO2)
# =========================
df_2025 = df[df["Year"] == 2025].copy()

# đảm bảo có NO2
df_2025 = df_2025.dropna(subset=["National_Urban_NO2_Sum"])
df_2025 = df_2025[df_2025["National_Urban_NO2_Sum"] > 0]

# log NO2
df_2025["log_NO2"] = np.log(df_2025["National_Urban_NO2_Sum"])

# =========================
# 4. PREDICT GDP 2025
# =========================

# predict log GDP
df_2025["log_GDP_pred"] = beta0 + beta1 * df_2025["log_NO2"]

# convert về GDP thật
df_2025["GDP_pred"] = np.exp(df_2025["log_GDP_pred"])

# =========================
# 5. SAVE & OUTPUT
# =========================
result = df_2025[["ISO", "Country_x", "National_Urban_NO2_Sum", "GDP_pred"]]

print("\n===== GDP 2025 PREDICTIONS =====")
print(result.head())

# save file
result.to_csv("gdp_2025_predictions.csv", index=False)

print("\n✅ Saved: gdp_2025_predictions.csv")