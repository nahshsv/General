import pandas as pd

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("csv/city_total_NO2_2005_2025.csv")

# ===============================
# GROUP BY COUNTRY + YEAR
# ===============================
country_df = (
    df.groupby(["Year", "ISO", "Country"], as_index=False)
      .agg({"Total_NO2_molecules": "sum"})
)

# đổi tên column cho giống format bạn muốn
country_df = country_df.rename(columns={
    "Total_NO2_molecules": "National_Urban_NO2_Sum"
})

# ===============================
# SORT (optional nhưng nên có)
# ===============================
country_df = country_df.sort_values(["ISO", "Year"])

# ===============================
# SAVE
# ===============================
country_df.to_csv("country_NO2_2005_2025.csv", index=False)

print(country_df.head())