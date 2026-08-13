import pandas as pd

# ==========================================
# LOAD FILES
# ==========================================

no2_df = pd.read_csv("csv/country_year_total_NO2.csv")

gdp_df = pd.read_csv("csv/gdp_africa_2005_2024.csv")

# ==========================================
# KEEP ONLY 2018+
# ==========================================

gdp_df = gdp_df[gdp_df["Year"] >= 2018]

# ==========================================
# MERGE USING ISO + YEAR ONLY
# ==========================================

merged_df = pd.merge(
    no2_df,
    gdp_df,
    on=["ISO", "Year"],
    how="left"
)

# ==========================================
# SORT
# ==========================================

merged_df = merged_df.sort_values(
    by=["ISO", "Year"]
)

# ==========================================
# SAVE
# ==========================================

merged_df.to_csv(
    "merged_NO2_GDP_2018_2025.csv",
    index=False
)

print("Saved: merged_NO2_GDP_2018_2025.csv")

print(merged_df.head())