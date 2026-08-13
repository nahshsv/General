import sys

import xarray as xr
import numpy as np
import glob
from tqdm import tqdm

YEAR = sys.argv[1]
DATA_DIR = "omi_daily"
OUTPUT_FILE = f"omi_annual/Africa_NO2_{YEAR}_annual.nc"

VAR = "ColumnAmountNO2TropCloudScreened"
LAT_MIN = -40
LAT_MAX = 40
LON_MIN = -20
LON_MAX = 55

files = sorted(glob.glob(f"{DATA_DIR}/*OMNO2d_{YEAR}*.he5"))

# create global OMI grid
lat = np.arange(-89.875, 90, 0.25)
lon = np.arange(-179.875, 180, 0.25)

data_list = []

for f in tqdm(files):

    ds = xr.open_dataset(
        f,
        group="HDFEOS/GRIDS/ColumnAmountNO2/Data Fields",
        engine="netcdf4"
    )

    no2 = ds[VAR]

    # attach coordinates
    no2 = xr.DataArray(
        no2.values,
        coords=[lat, lon],
        dims=["lat", "lon"]
    )

    # remove fill value
    no2 = no2.where(no2 > -1e20)

    # subset africa
    no2 = no2.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    data_list.append(no2)

    ds.close()


stack = xr.concat(data_list, dim="time")

annual = stack.mean(dim="time")

annual.to_netcdf(OUTPUT_FILE)

print("Saved:", OUTPUT_FILE)