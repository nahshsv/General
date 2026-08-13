import xarray as xr
import pandas as pd
import numpy as np
import os

# ===============================
# CONFIGURATION
# ===============================
CITY_FILE = "csv/africa_major_cities_NO2_dataset_complete.csv"
DATA_DIR = "omi_annual/"  # Đã khớp với ảnh của bạn
OUTPUT_FILE = "africa_cities_NO2_timeseries.csv"

# Map Tier với bán kính (km)
TIER_MAP = {1: 50, 2: 25, 3: 10}
EARTH_RADIUS = 6371000

# Giới hạn Châu Phi để crop cho nhẹ máy
LAT_MIN, LAT_MAX = -35, 37
LON_MIN, LON_MAX = -20, 55

# ===============================
# LOAD CITIES
# ===============================
if not os.path.exists(CITY_FILE):
    print(f"❌ Không tìm thấy file CSV: {CITY_FILE}")
    exit()

cities = pd.read_csv(CITY_FILE)
print(f"✅ Loaded {len(cities)} cities.")

# Lấy danh sách các file .nc từ folder data_annual/
if not os.path.exists(DATA_DIR):
    print(f"❌ Thư mục {DATA_DIR} không tồn tại!")
    exit()

no2_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.nc')])
print(f"📂 Found {len(no2_files)} NetCDF files.")

all_year_results = []

# ===============================
# PROCESS EACH FILE (YEAR)
# ===============================
for file_name in no2_files:
    # Logic tách năm từ: NO2_annual_2005.nc
    try:
        year = file_name.split('_')[2].split('.')[0] 
    except IndexError:
        print(f"⚠️ Không thể tách năm từ file: {file_name}")
        continue

    print(f"🚀 Processing Year: {year}...", end="\r")
    
    file_path = os.path.join(DATA_DIR, file_name)
    ds = xr.open_dataset(file_path)
    
    # Lấy data và crop sơ bộ vùng Châu Phi
    no2_full = ds["__xarray_dataarray_variable__"].sel(
        lat=slice(LAT_MIN, LAT_MAX), 
        lon=slice(LON_MIN, LON_MAX)
    )
    
    # Tính độ phân giải grid (rad)
    dlat = np.deg2rad(abs(float(ds.lat[1] - ds.lat[0])))
    dlon = np.deg2rad(abs(float(ds.lon[1] - ds.lon[0])))

    for i, row in cities.iterrows():
        # Lấy bán kính theo Tier, mặc định 10km nếu không có tier
        buffer_km = TIER_MAP.get(row.get('Tier', 3), 10)
        buffer_deg = buffer_km / 111.0
        
        city_lat = row["Latitude"]
        city_lon = row["Longitude"]

        # Crop vùng buffer quanh thành phố
        region = no2_full.sel(
            lat=slice(city_lat - buffer_deg, city_lat + buffer_deg),
            lon=slice(city_lon - buffer_deg, city_lon + buffer_deg)
        )

        if region.size == 0:
            continue

        # Tính diện tích từng pixel (m2 -> cm2)
        lat_rad = np.deg2rad(region.lat)
        pixel_area_cm2 = (EARTH_RADIUS**2) * dlat * dlon * np.cos(lat_rad) * 1e4

        # Broadcast diện tích để nhân ma trận
        area_2d = xr.DataArray(pixel_area_cm2, coords={"lat": region.lat}, dims=["lat"]).broadcast_like(region)

        # Tính tổng khối lượng (Total molecules)
        total_no2 = float((region * area_2d).sum(skipna=True).values)

        all_year_results.append({
            "Year": int(year),
            "ISO": row["ISO"],
            "Country": row["Country"],
            "City": row["City"],
            "Tier": row.get('Tier', 3),
            "Buffer_KM": buffer_km,
            "Total_NO2_molecules": total_no2
        })
    
    ds.close() # Đóng file để giải phóng tài nguyên

# ===============================
# SAVE RESULTS
# ===============================
if all_year_results:
    final_df = pd.DataFrame(all_year_results)
    # Sắp xếp lại cho đẹp: theo quốc gia rồi tới năm
    final_df = final_df.sort_values(by=["Country", "City", "Year"])
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✨ Success! Saved results to {OUTPUT_FILE}")
else:
    print("\n❌ Không có dữ liệu nào được xử lý.")