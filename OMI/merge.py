import pandas as pd

# LOAD
gdp = pd.read_csv("csv/gdp_africa_2005_2024.csv")
no2 = pd.read_csv("csv/africa_no2_minus_nox_long_k=0.7.csv")

# MERGE (KEEP ALL)
merged = pd.merge(
    no2,
    gdp,
    on=["ISO", "Year"],
    how="outer"
)

# FIX COUNTRY COLUMN
merged["Country"] = merged["Country_x"].combine_first(merged["Country_y"])
merged = merged.drop(columns=["Country_x", "Country_y"])

# SORT
merged = merged.sort_values(by=["ISO", "Year"])

# SAVE (EMPTY INSTEAD OF NaN)
merged.to_csv(
    "africa_gdp_no2_merged.csv",
    index=False,
    na_rep=""
)

print("Done clean merge!")