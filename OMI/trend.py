import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# SETTINGS
# ===============================
k_values = [0.2, 0.5, 0.8]   # 👈 đổi tùy ý
data_path = "csv_merge"

# ===============================
# LOAD MAP (CHỈ LOAD 1 LẦN)
# ===============================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)

iso_list = [
"AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM",
"COD","COG","CIV","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA",
"GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT","MUS",
"MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
"ZAF","SDS","SDN","TZA","TGO","UGA","ZMB","ZWE"
]

base_map = world[world["ADM0_A3"].isin(iso_list)].copy()
base_map = base_map.to_crs("EPSG:4326")

# ===============================
# FUNCTION: CALCULATE TREND
# ===============================
def compute_trend(df):
    trend_list = []

    for iso, group in df.groupby("ISO"):
        group = group.sort_values("Year")

        if len(group) < 2:
            continue

        start = group.iloc[0]["National_Urban_NO2_Sum"]
        end = group.iloc[-1]["National_Urban_NO2_Sum"]

        trend = end - start

        trend_list.append({
            "ISO": iso,
            "trend": trend
        })

    trend_df = pd.DataFrame(trend_list)

    # normalize
    max_abs = np.max(np.abs(trend_df["trend"]))
    trend_df["trend_norm"] = trend_df["trend"] / max_abs

    return trend_df

# ===============================
# CREATE SUBPLOTS
# ===============================
fig, axes = plt.subplots(1, len(k_values), figsize=(16, 6))

# nếu chỉ 1 plot thì tránh lỗi
if len(k_values) == 1:
    axes = [axes]

# ===============================
# LOOP QUA k
# ===============================
for ax, k in zip(axes, k_values):

    file_name = f"{data_path}/africa_gdp_no2_merged_k={k:.1f}.csv"
    df = pd.read_csv(file_name)

    trend_df = compute_trend(df)

    # merge
    countries = base_map.merge(trend_df, left_on="ADM0_A3", right_on="ISO", how="left")

    # plot (KHÔNG legend ở đây)
    countries.plot(
        column="trend_norm",
        cmap="RdBu_r",
        linewidth=0.5,
        edgecolor="black",
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title(f"k = {k}", fontsize=12)
    ax.axis("off")

# ===============================
# ADD SHARED COLORBAR
# ===============================
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
sm._A = []

cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
cbar.set_label("Normalized Trend (Blue = Decrease, Red = Increase)")

# ===============================
# TITLE
# ===============================
fig.suptitle("Trend of NO₂ (2005–2025) for Different k Values", fontsize=14)

# ===============================
# SAVE
# ===============================
plt.savefig("no2_trend_comparison.png", dpi=300, bbox_inches="tight")

plt.show()