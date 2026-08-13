import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import regionmask
import glob
import matplotlib.pyplot as plt
import os
import re

# =====================================
# CONFIG
# =====================================
DATA_DIR = "omi_annual/"
FILE_PATTERN = DATA_DIR + "Africa_NO2_*_annual.nc"
VAR_NAME = "__xarray_dataarray_variable__"

URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

# ⚠️ FIX typo: SDS -> SSD
iso_list = [
"AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM",
"COD","COG","CIV","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA",
"GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT","MUS",
"MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
"ZAF","SDS","SDN","TZA","TGO","UGA","ZMB","ZWE"
]

# =====================================
# 1️⃣ LOAD NC FILES
# =====================================
files = sorted(glob.glob(FILE_PATTERN))
print(f"📂 Found {len(files)} files")

datasets = []
years = []

for f in files:
    filename = os.path.basename(f)

    match = re.search(r"\d{4}", filename)
    if not match:
        print(f"⚠️ Skip file: {filename}")
        continue

    year = int(match.group())

    ds = xr.open_dataset(f)

    if VAR_NAME not in ds:
        raise ValueError(f"❌ Variable not found in {filename}")

    da = ds[VAR_NAME].squeeze()

    datasets.append(da)
    years.append(year)

print(f"✅ Loaded {len(datasets)} datasets")

da_all = xr.concat(datasets, dim="time")
da_all["time"] = years

# =====================================
# 2️⃣ TIME AVERAGE
# =====================================
da_mean = da_all.mean(dim="time", skipna=True)

# =====================================
# 3️⃣ LOAD SHAPE
# =====================================
world = gpd.read_file(URL)

africa = world[world["ADM0_A3"].isin(iso_list)].copy().reset_index(drop=True)

print(f"🌍 Countries: {len(africa)}")

# =====================================
# 4️⃣ MASK
# =====================================
mask = regionmask.mask_geopandas(
    africa,
    da_mean.lon,
    da_mean.lat
)

# =====================================
# 5️⃣ COUNTRY MEAN
# =====================================
values = []

for i in range(len(africa)):
    iso = africa.iloc[i]["ADM0_A3"]

    country_mask = mask == i
    data = da_mean.where(country_mask)

    mean_val = data.mean().item()

    if np.isnan(mean_val):
        lat = africa.iloc[i].geometry.centroid.y
        lon = africa.iloc[i].geometry.centroid.x

        mean_val = da_mean.sel(lat=lat, lon=lon, method="nearest").item()
        print(f"⚠️ Nearest used: {iso}")

    values.append(mean_val)

africa["NO2_mean"] = values

# =====================================
# 6️⃣ REMOVE OUTLIER (CLIP)
# =====================================
q99 = africa["NO2_mean"].quantile(0.99)

print("📊 99th percentile:", q99)

africa["NO2_mean_clipped"] = np.clip(
    africa["NO2_mean"],
    None,
    q99
)

# =====================================
# NORMALIZE (SAU KHI CLIP)
# =====================================
min_val = africa["NO2_mean_clipped"].min()
max_val = africa["NO2_mean_clipped"].max()

africa["NO2_norm"] = (
    africa["NO2_mean_clipped"] - min_val
) / (max_val - min_val)

# =====================================
# 7️⃣ SAVE CSV
# =====================================
df_out = africa[["ADM0_A3", "NO2_mean", "NO2_norm"]]
df_out.to_csv("Africa_NO2_normalized.csv", index=False)

print("✅ CSV saved")

# =====================================
# 8️⃣ MERGE
# =====================================
gdf = world.merge(df_out, on="ADM0_A3", how="left")
gdf = gdf[gdf["ADM0_A3"].isin(iso_list)]

world_bg = world[world["ADM0_A3"].isin(iso_list)]

# =====================================
# 9️⃣ PLOT (COLORBAR BÊN TRÁI)
# =====================================
fig, ax = plt.subplots(figsize=(12, 8))

# 👉 kéo map sang trái
ax.set_position([0.2, 0.1, 0.75, 0.8])

# nền
world_bg.plot(
    ax=ax,
    color="#f0f0f0",
    edgecolor="white",
    linewidth=0.1
)

# 👉 colorbar sát hơn
cax = fig.add_axes([0.1, 0.25, 0.02, 0.5])

gdf.plot(
    column="NO2_norm",
    cmap="Reds",
    vmin=0,
    vmax=1,
    legend=True,
    cax=cax,
    ax=ax,
    edgecolor="black",
    linewidth=0.4
)

cax.set_title("NO$_2$\n(0–1)", fontsize=10)

ax.set_title("Normalized NO$_2$ (Sub-Saharan Africa, 2005–2025)", fontsize=15)
ax.axis("off")

plt.show()