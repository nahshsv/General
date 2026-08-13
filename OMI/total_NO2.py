import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import regionmask
import os

# =====================================
# CONFIG
# =====================================
DATA_DIR = "omi_annual/"
OUTPUT_FILE = "Africa_NO2_Total_Molecules.csv"

var_name = "__xarray_dataarray_variable__"
R = 6371000  # Earth radius (m)

# =====================================
# LOAD COUNTRIES
# =====================================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)

iso_list = [
"AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM",
"COD","COG","CIV","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA",
"GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT","MUS",
"MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
"ZAF","SDS","SDN","TZA","TGO","UGA","ZMB","ZWE"
]

countries = world[world["ADM0_A3"].isin(iso_list)].copy()
countries = countries.to_crs("EPSG:4326").reset_index(drop=True)

# =====================================
# CREATE REGION MASK
# =====================================
regions = regionmask.from_geopandas(
    countries,
    names="NAME",
    abbrevs="ADM0_A3"
)

# =====================================
# CENTROIDS (for fallback)
# =====================================
countries["centroid"] = countries.geometry.centroid
centroids_lat = countries["centroid"].y.values
centroids_lon = countries["centroid"].x.values

# =====================================
# STORAGE
# =====================================
all_data = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".nc")])

# =====================================
# LOOP YEARS
# =====================================
for file in files:

    year = int(file.split("_")[2])
    print(f"Processing {year}...")

    ds = xr.open_dataset(os.path.join(DATA_DIR, file))
    no2 = ds[var_name]

    # =====================================
    # AREA CALCULATION
    # =====================================
    dlat = np.deg2rad(abs(ds.lat[1] - ds.lat[0]))
    dlon = np.deg2rad(abs(ds.lon[1] - ds.lon[0]))

    lat_rad = np.deg2rad(ds.lat)
    area = (R**2) * dlat * dlon * np.cos(lat_rad)
    area_2D = area.broadcast_like(no2)

    # =====================================
    # DAYS IN YEAR
    # =====================================
    days = 366 if year % 4 == 0 else 365

    # =====================================
    # MOLECULES GRID
    # =====================================
    molecules_grid = no2 * area_2D * 1e4 * days

    # =====================================
    # CREATE MASK
    # =====================================
    mask = regions.mask(no2)

    # =====================================
    # STACK + GROUPBY
    # =====================================
    mol_stack = molecules_grid.stack(grid=("lat", "lon"))
    mask_stack = mask.stack(grid=("lat", "lon"))

    country_sum = mol_stack.groupby(mask_stack).sum("grid")

    # =====================================
    # SAVE EACH COUNTRY
    # =====================================
    for idx in range(len(regions.names)):

        iso = regions.abbrevs[idx]
        name = regions.names[idx]

        if idx in country_sum.mask.values:
            val = float(country_sum.sel(mask=idx).values)

        else:
            # =====================================
            # FALLBACK: NEAREST GRID CELL
            # =====================================
            lat_c = centroids_lat[idx]
            lon_c = centroids_lon[idx]

            # find nearest grid index
            lat_idx = np.abs(ds.lat - lat_c).argmin().item()
            lon_idx = np.abs(ds.lon - lon_c).argmin().item()

            try:
                val = float(molecules_grid.isel(lat=lat_idx, lon=lon_idx).values)
            except:
                val = 0.0

        all_data.append({
            "ISO_A3": iso,
            "Country": name,
            "Year": year,
            "Total_NO2": val
        })

    ds.close()

# =====================================
# FINAL TABLE (WIDE FORMAT)
# =====================================
df = pd.DataFrame(all_data)

pivot = df.pivot(index=["ISO_A3", "Country"], columns="Year", values="Total_NO2")
pivot = pivot.reset_index()

pivot.to_csv(OUTPUT_FILE, index=False)

print("✅ Done:", OUTPUT_FILE)