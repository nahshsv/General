import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# FILE PATH
# ===============================
file1 = "Africa_NO2_Total_Molecules.csv"
file2 = "Africa_NO2_Total_Molecules_Satellite.csv"

# ===============================
# LOAD DATA
# ===============================
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("DF1 shape:", df1.shape)
print("DF2 shape:", df2.shape)

# ===============================
# CLEAN COUNTRY (optional)
# ===============================
df1["Country"] = df1["Country"].str.strip()
df2["Country"] = df2["Country"].str.strip()

# ===============================
# MERGE USING ISO (IMPORTANT)
# ===============================
df = df1.merge(df2, on="ISO_A3", how="inner", suffixes=("_A", "_B"))

print("\nAfter merge:", df.shape)

# ===============================
# FIX COUNTRY COLUMN
# ===============================
df["Country"] = df["Country_A"]
df = df.drop(columns=["Country_A", "Country_B"])

# ===============================
# GET YEAR COLUMNS
# ===============================
years = [c.replace("_A", "") for c in df.columns if c.endswith("_A")]

print("\nYears detected:", years)

# ===============================
# CALCULATE % DIFFERENCE
# ===============================
for y in years:
    df[f"{y}_pct_diff"] = np.where(
        df[f"{y}_B"] == 0,
        np.nan,
        (df[f"{y}_A"] - df[f"{y}_B"]) / df[f"{y}_B"] * 100
    )

# ===============================
# AVERAGE % DIFFERENCE
# ===============================
df["avg_pct_diff"] = df[[f"{y}_pct_diff" for y in years]].mean(axis=1)

print("\n📊 Average % difference per country:")
print(df[["ISO_A3", "Country", "avg_pct_diff"]].head())

# ===============================
# GLOBAL AVERAGE
# ===============================
global_avg = df["avg_pct_diff"].mean()
print(f"\n🌍 Global average difference: {global_avg:.2f}%")

# ===============================
# SELECT COUNTRIES TO PLOT
# ===============================
selected_countries = ["NGA", "ZAF", "KEN", "ETH"]  # bạn đổi tùy ý

plot_df = df[df["ISO_A3"].isin(selected_countries)]

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(10,6))

for _, row in plot_df.iterrows():
    values = [row[f"{y}_pct_diff"] for y in years]
    plt.plot(years, values, marker='o', label=row["ISO_A3"])

plt.title("Percentage Difference between NO2 Datasets")
plt.xlabel("Year")
plt.ylabel("% Difference")
plt.legend(title="ISO Country")
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()