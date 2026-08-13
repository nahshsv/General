import pandas as pd

# đọc file
no2 = pd.read_csv("csv/Africa_NO2_Total_Molecules.csv")
gdp = pd.read_csv("csv/gdp_africa_2005_2024.csv")

# bỏ Country ở GDP để tránh conflict
gdp = gdp.drop(columns=["Country"])

# melt NO2
no2_long = no2.melt(
    id_vars=["ISO_A3", "Country"],
    var_name="Year",
    value_name="National_Urban_NO2_Sum"
)

no2_long = no2_long.rename(columns={"ISO_A3": "ISO"})
no2_long["Year"] = no2_long["Year"].astype(int)

# merge
df = pd.merge(no2_long, gdp, on=["ISO", "Year"], how="left")

# chọn cột
df = df[[
    "Year",
    "ISO",
    "National_Urban_NO2_Sum",
    "GDP",
    "log_GDP",
    "Country"
]]

# sort
df = df.sort_values(["ISO", "Year"])

# save
df.to_csv("africa_gdp_no2_merged.csv", index=False)

print(df.head())