import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("africa_gdp_no2_merged.csv")
df = df.dropna(subset=["National_Urban_NO2_Sum"])

# ===============================
# AVERAGE NO2
# ===============================
avg_no2 = df.groupby("ISO")["National_Urban_NO2_Sum"].mean().reset_index()
avg_no2.rename(columns={"National_Urban_NO2_Sum": "NO2_mean"}, inplace=True)

# ===============================
# TREND (LOG)
# ===============================
def compute_trend(group):
    group = group.sort_values("Year")
    
    if len(group) < 2:
        return np.nan
    
    x = group["Year"].values
    y = np.log(group["National_Urban_NO2_Sum"].values)
    
    slope = np.polyfit(x, y, 1)[0]
    return slope

trend_no2 = df.groupby("ISO").apply(compute_trend).reset_index()
trend_no2.columns = ["ISO", "NO2_trend"]

# ===============================
# MERGE
# ===============================
data = avg_no2.merge(trend_no2, on="ISO")

# ===============================
# NORMALIZE
# ===============================
data["NO2_mean_norm"] = (
    data["NO2_mean"] - data["NO2_mean"].min()
) / (data["NO2_mean"].max() - data["NO2_mean"].min())

max_abs = data["NO2_trend"].abs().max()
data["NO2_trend_norm"] = data["NO2_trend"] / max_abs

# ===============================
# LOAD MAP (THEO CÁCH BẠN MUỐN)
# ===============================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)

# chọn Africa
africa = world[world["CONTINENT"] == "Africa"]

# rename để match ISO
africa = africa.rename(columns={"ADM0_A3": "ISO"})

# ===============================
# MERGE MAP + DATA
# ===============================
map_data = africa.merge(data, on="ISO", how="left")

# ===============================
# PLOT
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- LEFT: AVERAGE ----
map_data.plot(
    column="NO2_mean_norm",
    cmap="Reds",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Normalized Total NO₂ (0–1)")
axes[0].axis("off")

# ---- RIGHT: TREND ----
map_data.plot(
    column="NO2_trend_norm",
    cmap="RdBu_r",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    ax=axes[1]
)
axes[1].set_title("Normalized NO₂ Trend (-1 to 1)\nBlue=Decrease | Red=Increase")
axes[1].axis("off")

plt.tight_layout()
plt.show()