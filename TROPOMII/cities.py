import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask

# ==========================================
# INPUTS
# ==========================================

CITY_FILE = "africa_major_cities_NO2_dataset_complete.csv"

DATA_FOLDER = "data"

YEARS = range(2018, 2026)

BUFFER_KM = 30
BUFFER_M = BUFFER_KM * 1000

AVOGADRO = 6.02214076e23

EARTH_RADIUS = 6371000  # meters

# ==========================================
# LOAD CITY DATA
# ==========================================

cities = pd.read_csv(CITY_FILE)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    cities,
    geometry=gpd.points_from_xy(
        cities.Longitude,
        cities.Latitude
    ),
    crs="EPSG:4326"
)

# ==========================================
# CREATE TRUE 30KM BUFFERS
# ==========================================

# Convert to metric CRS
gdf_metric = gdf.to_crs(epsg=3857)

# Create 30km radius buffers
gdf_metric["geometry"] = gdf_metric.buffer(BUFFER_M)

# ==========================================
# PROCESS YEARS
# ==========================================

results = []

for year in YEARS:

    print(f"\nProcessing {year}...")

    raster_path = f"{DATA_FOLDER}/africa_no2_{year}.tif"

    # ==========================================
    # OPEN RASTER
    # ==========================================

    src = rasterio.open(raster_path)

    print("Raster CRS:", src.crs)

    # Convert buffers to raster CRS
    gdf_final = gdf_metric.to_crs(src.crs)

    # Pixel size
    pixel_width = src.res[0]
    pixel_height = abs(src.res[1])

    print("Pixel Resolution:", src.res)

    # ==========================================
    # PROCESS EACH CITY
    # ==========================================

    for idx, row in gdf_final.iterrows():

        iso = row["ISO"]
        country = row["Country"]
        city = row["City"]

        try:

            # ==========================================
            # MASK RASTER TO BUFFER
            # ==========================================

            out_image, out_transform = mask(
                src,
                [row.geometry],
                crop=True
            )

            # First band
            no2 = out_image[0]

            # Remove nodata
            no2 = no2.astype(float)

            if src.nodata is not None:
                no2[no2 == src.nodata] = np.nan

            # ==========================================
            # CONVERT mol/m² -> molecules/cm²
            # ==========================================

            no2_molecules_cm2 = no2 * 6.02214076e19

            # ==========================================
            # COMPUTE PIXEL AREA
            # ==========================================

            # Approx pixel area in m²
            pixel_area_m2 = pixel_width * pixel_height

            # Convert to cm²
            pixel_area_cm2 = pixel_area_m2 * 1e4

            # ==========================================
            # TOTAL NO2
            # ==========================================

            total_no2 = np.nansum(
                no2_molecules_cm2 * pixel_area_cm2
            )

            results.append({
                "ISO": iso,
                "Country": country,
                "City": city,
                "Year": year,
                "Total_NO2_molecules": total_no2
            })

        except Exception as e:

            print(f"Error processing {city} ({year}): {e}")

# ==========================================
# CREATE DATAFRAMES
# ==========================================

city_year_df = pd.DataFrame(results)

# ==========================================
# AGGREGATE COUNTRY-YEAR
# ==========================================

country_year_df = (
    city_year_df
    .groupby(["ISO", "Country", "Year"])["Total_NO2_molecules"]
    .sum()
    .reset_index()
)

# ==========================================
# SAVE OUTPUTS
# ==========================================

city_year_df.to_csv(
    "city_year_total_NO2.csv",
    index=False
)

country_year_df.to_csv(
    "country_year_total_NO2.csv",
    index=False
)

print("\nDone!")
print(country_year_df.head())