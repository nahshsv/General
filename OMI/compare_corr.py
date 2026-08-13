import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ===============================
# SETTINGS
# ===============================
k_values = np.arange(0.1, 1.0, 0.1)

corr_results = []

# ===============================
# LOOP QUA TỪNG FILE
# ===============================
for k in k_values:
    file_name = f"csv_merge/africa_gdp_no2_merged_k={k:.1f}.csv"
    
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Missing file: {file_name}")
        continue

    # bỏ NA
    df = df.dropna(subset=["National_Urban_NO2_Sum", "log_GDP"])

    # tránh log(0)
    df = df[df["National_Urban_NO2_Sum"] > 0]

    # log NO2
    df["log_NO2"] = np.log(df["National_Urban_NO2_Sum"])

    # ===============================
    # OPTION 1: GLOBAL CORRELATION
    # ===============================
    corr_raw, _ = pearsonr(df["log_NO2"], df["log_GDP"])

    # ===============================
    # OPTION 2 (khuyên dùng): MEAN BY COUNTRY
    # ===============================
    # corrs = []
    # for iso, g in df.groupby("ISO"):
    #     g = g[g["National_Urban_NO2_Sum"] > 0]
    #     if len(g) > 2:
    #         g["log_NO2"] = np.log(g["National_Urban_NO2_Sum"])
    #         r, _ = pearsonr(g["log_NO2"], g["log_GDP"])
    #         corrs.append(r)
    #
    # if len(corrs) > 0:
    #     corr_raw = np.mean(corrs)
    # else:
    #     corr_raw = np.nan

    # round để hiển thị
    corr = round(corr_raw, 4)

    corr_results.append({
        "k": round(k, 1),
        "correlation": corr
    })

# ===============================
# TẠO DATAFRAME
# ===============================
corr_df = pd.DataFrame(corr_results)

print("\nCorrelation table:")
print(corr_df)

# ===============================
# FIND BEST k
# ===============================
best_row = corr_df.loc[corr_df["correlation"].idxmax()]

print("\nBest k:", best_row["k"])
print("Max correlation:", best_row["correlation"])

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(8, 5))

plt.scatter(corr_df["k"], corr_df["correlation"], s=60)
plt.plot(corr_df["k"], corr_df["correlation"])

plt.xlabel("k value")
plt.ylabel("Pearson Correlation (log NO₂ vs log GDP)")
plt.title("Correlation vs k")

# highlight best k
plt.scatter(best_row["k"], best_row["correlation"], s=120)
plt.text(best_row["k"], best_row["correlation"],
         f"  k={best_row['k']:.1f}, r={best_row['correlation']:.4f}",
         fontsize=10)

plt.grid(True,alpha=0.3)

# ===============================
# SAVE FIGURE
# ===============================
plt.savefig("correlation_vs_k.png", dpi=300, bbox_inches="tight")

plt.show()