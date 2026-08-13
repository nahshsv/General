import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1. SETTINGS
# ===============================
SELECTED_YEARS = list(range(2005, 2026))

TITLE_PERIOD = f"{SELECTED_YEARS[0]}-{SELECTED_YEARS[-1]}" if len(SELECTED_YEARS) > 1 else f"{SELECTED_YEARS[0]}"

# ===============================
# 2. LOAD & FILTER DATA
# ===============================
no2_df = pd.read_csv("csv/africa_national_urban_NO2_summary.csv")

# Lọc năm
filtered_df = no2_df[no2_df['Year'].isin(SELECTED_YEARS)]

# ❌ BỎ đoạn này (vì không còn city nữa)
# country_yearly = ...

# ✅ Chỉ cần lấy mean theo quốc gia
country_map_data = (
    filtered_df
    .groupby(['ISO', 'Country'])['National_Urban_NO2_Sum']
    .mean()
    .reset_index()
)

# ===============================
# 3. LOAD GEOGRAPHY & MERGE
# ===============================
url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
world = gpd.read_file(url)
africa = world[world["CONTINENT"] == "Africa"].copy()

africa = africa.merge(
    country_map_data,
    left_on="ADM0_A3",
    right_on="ISO",
    how="left"
)

# ===============================
# 4. PLOT
# ===============================
fig, ax = plt.subplots(figsize=(10, 12))

africa.plot(
    column="National_Urban_NO2_Sum",   # ✅ đổi tên cột
    cmap="YlOrRd",
    linewidth=0.6,
    edgecolor="0.4",
    legend=True,
    missing_kwds={"color": "#f0f0f0", "label": "No Data"},
    legend_kwds={
        "label": f"Average Urban NO₂ (molecules) - Period {TITLE_PERIOD}",
        "orientation": "horizontal",
        "shrink": 0.5,
        "pad": 0.05
    },
    ax=ax
)

ax.set_title(
    f"Satellite Urban NO₂ Concentration in Africa\n(Period: {TITLE_PERIOD})",
    fontsize=16,
    fontweight='bold',
    loc='center',   # ✅ đảm bảo center
    y=1.02          # ✅ đẩy lên cao chút
)

ax.axis("off")
plt.tight_layout()
plt.show()