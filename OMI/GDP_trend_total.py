import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("africa_gdp_no2_merged.csv")
df = df.dropna(subset=["GDP"])

# ===============================
# 2. GDP MEAN
# ===============================
gdp_mean = df.groupby("ISO")["GDP"].mean().reset_index()
gdp_mean.rename(columns={"GDP": "GDP_mean"}, inplace=True)

# ===============================
# 3. GDP TREND (LOG)
# ===============================
def compute_gdp_trend(group):
    group = group.sort_values("Year")
    
    if len(group) < 2:
        return np.nan
    
    x = group["Year"].values
    y = np.log(group["GDP"].values)  # LOG = growth rate
    
    slope = np.polyfit(x, y, 1)[0]
    return slope

gdp_trend = df.groupby("ISO").apply(compute_gdp_trend).reset_index()
gdp_trend.columns = ["ISO", "GDP_trend"]

# ===============================
# 4. MERGE
# ===============================
data = gdp_mean.merge(gdp_trend, on="ISO")

# ===============================
# 5. NORMALIZE
# ===============================
# level (0–1)
data["GDP_mean_norm"] = (
    data["GDP_mean"] - data["GDP_mean"].min()
) / (data["GDP_mean"].max() - data["GDP_mean"].min())

# trend (-1 → 1)
max_abs = data["GDP_trend"].abs().max()
data["GDP_trend_norm"] = data["GDP_trend"] / max_abs

# ===============================
# 6. LOAD MAP (ADM0_A3)
# ===============================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)

africa = world[world["CONTINENT"] == "Africa"]
africa = africa.rename(columns={"ADM0_A3": "ISO"})

# ===============================
# 7. MERGE MAP
# ===============================
map_data = africa.merge(data, on="ISO", how="left")

# ===============================
# 8. PLOT
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- LEFT: GDP LEVEL ----
map_data.plot(
    column="GDP_mean_norm",
    cmap="Reds",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Normalized GDP Level (0–1)")
axes[0].axis("off")

# ---- RIGHT: GDP TREND ----
map_data.plot(
    column="GDP_trend_norm",
    cmap="RdBu_r",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    ax=axes[1]
)
axes[1].set_title("GDP Trend (-1 to 1)\nBlue=Slow/Decline | Red=Fast Growth")
axes[1].axis("off")

plt.tight_layout()
plt.show()