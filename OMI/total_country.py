import pandas as pd

# 1. Load file kết quả bạn vừa chạy xong
df = pd.read_csv("csv/africa_cities_NO2_timeseries.csv")

# 2. Group by theo Quốc gia (ISO) và Năm, sau đó tính tổng
# Chúng ta sum cột Total_NO2_molecules
country_summary = df.groupby(['Year', 'ISO', 'Country'])['Total_NO2_molecules'].sum().reset_index()

# 3. Đổi tên cột cho rõ nghĩa
country_summary = country_summary.rename(columns={'Total_NO2_molecules': 'National_Urban_NO2_Sum'})

# 4. Sắp xếp lại cho dễ nhìn
country_summary = country_summary.sort_values(by=['Country', 'Year'])

# 5. Lưu ra file mới để làm việc với GDP
country_summary.to_csv("csv/africa_national_urban_NO2_summary.csv", index=False)

print("✅ Đã tổng hợp xong! File mới có dạng mỗi hàng là một quốc gia/năm.")
print(country_summary.head(10))