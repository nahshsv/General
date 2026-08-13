# =====================================
# COUNT COUNTRIES WITH INCREASING / DECREASING NO2
# =====================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# =====================================
# 1. LOAD DATA
# =====================================
file_path = "csv/Africa_Fire_NOx_Molecules_Wide_k=0.8.csv"   # <-- đổi path
df = pd.read_csv(file_path)

# =====================================
# 2. RESHAPE (WIDE -> LONG)
# =====================================
# Các cột năm
year_cols = [col for col in df.columns if col.isdigit()]

df_long = df.melt(
    id_vars=["ISO_A3", "Country"],
    value_vars=year_cols,
    var_name="Year",
    value_name="NO2"
)

# Convert kiểu dữ liệu
df_long["Year"] = df_long["Year"].astype(int)
df_long["NO2"] = pd.to_numeric(df_long["NO2"], errors="coerce")

# Drop NA
df_long = df_long.dropna(subset=["NO2"])

# Log transform (quan trọng)
df_long["NO2_log"] = np.log(df_long["NO2"])

# =====================================
# 3. FUNCTION TREND
# =====================================
def get_trend(group):
    if group["Year"].nunique() < 3:
        return pd.Series({"slope": np.nan, "pval": np.nan})
    
    X = sm.add_constant(group["Year"])
    y = group["NO2_log"]
    
    model = sm.OLS(y, X).fit()
    
    return pd.Series({
        "slope": model.params["Year"],
        "pval": model.pvalues["Year"]
    })

# =====================================
# 4. APPLY CHO MỖI COUNTRY
# =====================================
trend_df = df_long.groupby("ISO_A3").apply(get_trend).reset_index()

# =====================================
# 5. PHÂN LOẠI
# =====================================
conditions = [
    (trend_df["pval"] < 0.05) & (trend_df["slope"] > 0),
    (trend_df["pval"] < 0.05) & (trend_df["slope"] < 0),
]

choices = ["Increasing", "Decreasing"]

trend_df["trend"] = np.select(conditions, choices, default="No trend")

# =====================================
# 6. COUNT
# =====================================
print("\n=== NO2 TREND COUNT ===")
print(trend_df["trend"].value_counts())

# =====================================
# 7. OPTIONAL: LIST COUNTRIES
# =====================================
print("\nCountries Increasing NO2:")
print(trend_df[trend_df["trend"] == "Increasing"]["ISO_A3"].tolist())

print("\nCountries Decreasing NO2:")
print(trend_df[trend_df["trend"] == "Decreasing"]["ISO_A3"].tolist())

# =====================================
# 8. SAVE
# =====================================
trend_df.to_csv("no2_trend_result.csv", index=False)