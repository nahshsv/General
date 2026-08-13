import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("csv/africa_gdp_no2_merged_k=0.8.csv")

# ===============================
# 2. COMPUTE AVERAGE PER COUNTRY
# ===============================
agg = df.groupby("ISO").agg({
    "National_Urban_NO2_Sum": "mean",
    "log_GDP": "mean"
}).reset_index()

# Rename
agg = agg.rename(columns={
    "National_Urban_NO2_Sum": "NO2_avg",
    "log_GDP": "GDP_avg"
})

# ===============================
# 3. NORMALIZE (0–1)
# ===============================
agg["NO2_norm"] = (
    (agg["NO2_avg"] - agg["NO2_avg"].min()) /
    (agg["NO2_avg"].max() - agg["NO2_avg"].min())
)

agg["GDP_norm"] = (
    (agg["GDP_avg"] - agg["GDP_avg"].min()) /
    (agg["GDP_avg"].max() - agg["GDP_avg"].min())
)

# ===============================
# 4. LOAD MAP
# ===============================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)
africa = world[world["CONTINENT"] == "Africa"].copy()

# Merge
africa = africa.merge(
    agg,
    left_on="ADM0_A3",
    right_on="ISO",
    how="left"
)

# Remove countries without data
africa = africa[africa["NO2_norm"].notna()]

# ===============================
# 5. CREATE FIGURE (2 MAPS)
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

cmap = "Reds"
norm = Normalize(vmin=0, vmax=1)

# ---- NO2 MAP ----
africa.plot(
    column="NO2_norm",
    cmap=cmap,
    linewidth=0.4,
    edgecolor="black",
    ax=axes[0],
    vmin=0,
    vmax=1
)
axes[0].set_title("Normalized NO₂", fontsize=12)
axes[0].axis("off")

# ---- GDP MAP ----
africa.plot(
    column="GDP_norm",
    cmap=cmap,
    linewidth=0.4,
    edgecolor="black",
    ax=axes[1],
    vmin=0,
    vmax=1
)
axes[1].set_title("Normalized GDP", fontsize=12)
axes[1].axis("off")

# ===============================
# 6. SHARED COLORBAR
# ===============================
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.035,   # 👈 giảm chiều cao
    pad=0.02,         # 👈 kéo lên gần map (quan trọng nhất)
    shrink=0.7,       # 👈 thu ngắn lại (đỡ dài)
    aspect=40         # 👈 giữ thanh mảnh
)

cbar.set_label("Normalized Value (0–1)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# ===============================
# 7. FINAL TOUCH
# ===============================
plt.suptitle(
    "Average Normalized NO₂ and GDP (Sub-Saharan Africa, 2005–2024)",
    fontsize=14
)

# Save high quality
plt.savefig(
    "NO2_GDP_side_by_side.png",
    dpi=300,           # dùng 600 nếu submit journal
    bbox_inches="tight"
)

plt.show()