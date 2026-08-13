import pandas as pd

# ===============================
# LOAD DATA
# ===============================
no2 = pd.read_csv("country_NO2_2005_2025.csv")
gdp = pd.read_csv("gdp_africa_2005_2024.csv")

print("NO2 columns:", no2.columns)
print("GDP columns:", gdp.columns)

# ===============================
# CLEAN DATA (optional nhưng nên có)
# ===============================

# đảm bảo kiểu dữ liệu đúng
no2["Year"] = no2["Year"].astype(int)
gdp["Year"] = gdp["Year"].astype(int)

# ===============================
# MERGE (FIX COUNTRY DUPLICATE)
# ===============================

df = pd.merge(
    no2,
    gdp.drop(columns=["Country"]),   # tránh Country_x, Country_y
    on=["ISO", "Year"],
    how="left"
)

# ===============================
# REORDER COLUMNS (GIỐNG HÌNH 3)
# ===============================

df = df[[
    "Year",
    "ISO",
    "National_Urban_NO2_Sum",
    "GDP",
    "log_GDP",
    "Country"
]]

# ===============================
# SORT
# ===============================

df = df.sort_values(["ISO", "Year"])

# ===============================
# CHECK MISSING (quan trọng)
# ===============================

print("\nMissing GDP rows:")
print(df[df["GDP"].isna()].head())

# ===============================
# SAVE
# ===============================

output_file = "merged_NO2_GDP.csv"
df.to_csv(output_file, index=False)

print("\nSaved file:", output_file)
print("Total rows:", len(df))