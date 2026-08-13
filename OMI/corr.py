import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("merged_NO2_GDP_total_cities.csv")

# =========================
# 2. CLEAN DATA
# =========================
df = df[['Year', 'ISO', 'National_Urban_NO2_Sum', 'GDP']]

df = df.rename(columns={
    'National_Urban_NO2_Sum': 'NO2'
})

# remove missing
df = df.dropna()

# remove duplicates (very important)
df = df.drop_duplicates(subset=['Year', 'ISO'])

# =========================
# 3. LOG TRANSFORM
# =========================
# tránh log(0)
df = df[df['GDP'] > 0]
df = df[df['NO2'] > 0]

df['log_GDP'] = np.log(df['GDP'])
df['log_NO2'] = np.log(df['NO2'])

# =========================
# 4. CROSS-SECTION CORRELATION
# =========================
results = []

for year in sorted(df['Year'].unique()):
    df_year = df[df['Year'] == year]

    # đảm bảo đủ sample
    if len(df_year) < 10:
        continue

    # Pearson (linear)
    pearson_corr = df_year['log_NO2'].corr(df_year['log_GDP'])

    # Spearman (rank)
    spearman_corr = df_year['log_NO2'].corr(df_year['log_GDP'], method='spearman')

    results.append({
        'Year': year,
        'Pearson_corr': pearson_corr,
        'Spearman_corr': spearman_corr,
        'N_countries': len(df_year)
    })

corr_df = pd.DataFrame(results)

print("\n===== CROSS-SECTION CORRELATION =====\n")
print(corr_df)

# =========================
# 5. SUMMARY STATS
# =========================
print("\n===== SUMMARY =====\n")
print("Average Pearson:", corr_df['Pearson_corr'].mean())
print("Std Pearson:", corr_df['Pearson_corr'].std())

print("Average Spearman:", corr_df['Spearman_corr'].mean())
print("Std Spearman:", corr_df['Spearman_corr'].std())

# =========================
# 6. PLOT
# =========================
plt.figure(figsize=(10,6))

plt.plot(corr_df['Year'], corr_df['Pearson_corr'], label='Pearson', marker='o')
plt.plot(corr_df['Year'], corr_df['Spearman_corr'], label='Spearman', marker='s')

plt.xlabel("Year")
plt.ylabel("Correlation")
plt.title("Cross-sectional Correlation: log(NO2) vs log(GDP)")
plt.legend()

plt.grid(True)
plt.show()