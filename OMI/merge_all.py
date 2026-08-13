import pandas as pd

# ===============================
# FILE PATHS
# ===============================
nox_file = "csv/Africa_Fire_NOx_Molecules_Wide_k=0.7.csv"
no2_file = "csv/Africa_NO2_Total_Molecules.csv"

# ===============================
# LOAD
# ===============================
nox = pd.read_csv(nox_file)
no2 = pd.read_csv(no2_file)

# ===============================
# WIDE → LONG
# ===============================
def to_long(df, value_name):
    df_long = df.melt(
        id_vars=["ISO_A3", "Country"],
        var_name="Year",
        value_name=value_name
    )
    df_long["Year"] = df_long["Year"].astype(int)
    return df_long

nox_long = to_long(nox, "NOx")
no2_long = to_long(no2, "NO2")

# ===============================
# MERGE (ISO ONLY)
# ===============================
df = pd.merge(
    nox_long,
    no2_long,
    on=["ISO_A3", "Year"],   # ❗ chỉ dùng ISO
    how="inner",
    suffixes=("_nox", "_no2")
)

# ===============================
# FIX COUNTRY
# ===============================
df["Country"] = df["Country_nox"]
df = df.drop(columns=["Country_nox", "Country_no2"])

# ===============================
# SUBTRACT (NO2 - NOx)
# ===============================
df["Value"] = df["NO2"] - df["NOx"]

# nếu muốn tránh âm:
# df["Value"] = df["Value"].clip(lower=0)

# ===============================
# FORMAT GIỐNG FILE CỦA BẠN
# ===============================
df_final = df[[
    "Year",
    "ISO_A3",
    "Country",
    "Value"
]]

df_final = df_final.rename(columns={
    "ISO_A3": "ISO",
    "Value": "National_Urban_NO2_Sum"
})

# ===============================
# SORT
# ===============================
df_final = df_final.sort_values(["ISO", "Year"])

# ===============================
# SAVE
# ===============================
output_file = "africa_no2_minus_nox_long.csv"
df_final.to_csv(output_file, index=False)

print("Saved:", output_file)