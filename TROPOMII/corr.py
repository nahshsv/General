import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("merged_NO2_GDP_2018_2025.csv")

# ==========================================
# DROP MISSING VALUES
# ==========================================

df = df.dropna(subset=["GDP", "Total_NO2_molecules"])

# ==========================================
# CREATE log_NO2
# ==========================================

df["log_NO2"] = np.log(df["Total_NO2_molecules"])

# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["log_NO2"],
    df["log_GDP"]
)

plt.xlabel("log(NO2)")
plt.ylabel("log(GDP)")
plt.title("log(NO2) vs log(GDP)")

plt.grid(True)

plt.show()

# ==========================================
# CORRELATION
# ==========================================

corr = df[["log_NO2", "log_GDP"]].corr()

print("\nCorrelation Matrix:\n")
print(corr)

# ==========================================
# OPTIONAL: PRINT SAMPLE
# ==========================================

print("\nSample Data:\n")

print(
    df[
        [
            "ISO",
            "Country_x",
            "Year",
            "Total_NO2_molecules",
            "log_NO2",
            "GDP",
            "log_GDP"
        ]
    ].head()
)