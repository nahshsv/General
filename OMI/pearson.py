import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("merged_NO2_GDP_total_cities.csv")

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna(subset=["Year", "National_Urban_NO2_Sum", "log_GDP"])
df = df[df["National_Urban_NO2_Sum"] > 0]

# =========================
# 3. LOG TRANSFORM NO2
# =========================
df["log_NO2"] = np.log(df["National_Urban_NO2_Sum"])

# =========================
# 4. PEARSON BY YEAR
# =========================
results = []

for year in sorted(df["Year"].unique()):
    subset = df[df["Year"] == year]
    
    if len(subset) > 3:
        r, p = pearsonr(subset["log_NO2"], subset["log_GDP"])
        results.append({
            "Year": year,
            "Correlation": r,
            "p_value": p
        })

corr_df = pd.DataFrame(results)

# =========================
# 5. PRINT RESULTS
# =========================
print("\n===== PEARSON BY YEAR =====")
print(corr_df)

# =========================
# 6. RANGE
# =========================
r_min = corr_df["Correlation"].min()
r_max = corr_df["Correlation"].max()
r_mean = corr_df["Correlation"].mean()

print("\n===== RANGE =====")
print(f"Min r: {r_min:.3f}")
print(f"Max r: {r_max:.3f}")
print(f"Mean r: {r_mean:.3f}")

# =========================
# 7. LINE CHART
# =========================
plt.figure(figsize=(10,5))

plt.plot(corr_df["Year"], corr_df["Correlation"], marker='o')

# mean line
plt.axhline(r_mean, linestyle='--', label=f"Mean = {r_mean:.2f}")

# show all years clearly
plt.xticks(corr_df["Year"], rotation=45)

plt.xlabel("Year", fontsize=12)
plt.ylabel("Pearson r", fontsize=12)
plt.title("Yearly Correlation between NO₂ and GDP", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

# save for poster
plt.savefig("pearson_by_year.png", dpi=300)

plt.show()

print("\n✅ Plot saved as 'pearson_by_year.png'")