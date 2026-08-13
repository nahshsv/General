import rasterio
import matplotlib.pyplot as plt

# Open GeoTIFF
src = rasterio.open("africa_no2_2025.tif")

# Read first band
no2 = src.read(1)

# Plot
plt.figure(figsize=(12, 8))

plt.imshow(no2)

plt.colorbar(label='NO2')

plt.title("Africa NO2 2020")

plt.show()