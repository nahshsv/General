import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("merged_NO2_GDP_2018_2025.csv")

# ==========================================
# CLEAN DATA
# ==========================================

df = df.dropna(subset=["GDP", "Total_NO2_molecules"])

# ==========================================
# OPTIONAL: CLEAN COUNTRY COLUMN
# ==========================================

if "Country_x" in df.columns:
    df = df.rename(columns={"Country_x": "Country"})

# ==========================================
# LOG TRANSFORM
# ==========================================

df["log_NO2"] = np.log(df["Total_NO2_molecules"])

# Create log_GDP if not exists
if "log_GDP" not in df.columns:
    df["log_GDP"] = np.log(df["GDP"])

# ==========================================
# DEFINE VARIABLES
# ==========================================

X = df["log_NO2"]

# Add intercept
X = sm.add_constant(X)

y = df["log_GDP"]

# ==========================================
# FIT OLS MODEL
# ==========================================

model = sm.OLS(y, X).fit()

# ==========================================
# PRINT RESULTS
# ==========================================

print("\nOLS Regression Results:\n")

print(model.summary())

# ==========================================
# PRINT INTERPRETATION
# ==========================================

beta = model.params["log_NO2"]

print("\n===================================")
print(f"Coefficient (beta): {beta:.4f}")
print("===================================\n")

print(
    f"A 1% increase in NO2 is associated "
    f"with approximately {beta:.2f}% "
    f"increase in GDP."
)

# ==========================================
# PLOT REGRESSION
# ==========================================

plt.figure(figsize=(8,6))

# Scatter
plt.scatter(
    df["log_NO2"],
    df["log_GDP"],
    alpha=0.7
)

# Predicted values
y_pred = model.predict(X)

# Sort values for smooth line
sorted_idx = np.argsort(df["log_NO2"])

plt.plot(
    df["log_NO2"].iloc[sorted_idx],
    y_pred.iloc[sorted_idx],
    linewidth=2
)

plt.xlabel("log(NO2)")
plt.ylabel("log(GDP)")
plt.title("OLS Regression: log(NO2) vs log(GDP)")

plt.grid(True)

plt.show()