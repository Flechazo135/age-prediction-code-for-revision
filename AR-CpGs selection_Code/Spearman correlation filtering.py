import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os
os.chdir(r"pathway")
data1 = pd.read_csv("GSE59509_removing missing.csv", header=0)
data2 = pd.read_csv("GSE119078_removing missing.csv", header=0)
data3 = pd.read_csv("GSE92767_removing missing.csv", header=0)
data4 = pd.read_csv("GSE111223_removing missing.csv", header=0)
data5 = pd.read_csv("GSE138279_removing missing.csv", header=0)
data6 = pd.read_csv("GSE78874_removing missing.csv", header=0)
df1 = data1.set_index('ID').transpose()
df2 = data2.set_index('ID').transpose()
df3 = data3.set_index('ID').transpose()
df4 = data4.set_index('ID').transpose()
df5 = data5.set_index('ID').transpose()
df6 = data6.set_index('ID').transpose()
# Obtain the common methylation sites
common_cpgs = set(df1.columns)
for df in [df2, df3, df4, df5, df6]:
    common_cpgs = common_cpgs.intersection(set(df.columns))
df1_common = df1[list(common_cpgs)]
df2_common = df2[list(common_cpgs)]
df3_common = df3[list(common_cpgs)]
df4_common = df4[list(common_cpgs)]
df5_common = df5[list(common_cpgs)]
df6_common = df6[list(common_cpgs)]
# Merge the datasets
combined_df = pd.concat([df1_common, df2_common, df3_common, df4_common, df5_common, df6_common], axis=0)
print(combined_df)
age = combined_df['age']
# Calculate Pearson and Spearman correlations
results = []
exclude_columns = ['age']
for cpg_site in combined_df.columns:
    if cpg_site not in exclude_columns:
        pearson_corr, pearson_p = pearsonr(combined_df[cpg_site], age)
        spearman_corr, spearman_p = spearmanr(combined_df[cpg_site], age)
        results.append({
            'CpG_Site': cpg_site,
            'Pearson_Correlation': pearson_corr,
            'Pearson_p-value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_p-value': spearman_p
        })
results_df = pd.DataFrame(results)
filtered_results = results_df[results_df['Spearman_Correlation'].abs() >= 0.6]# Select the CpG sites with Spearman correlation absolute values greater than or equal to 0.6
filtered_cpg_sites = filtered_results['CpG_Site'].tolist()
filtered_cpg_values = combined_df[['age'] + filtered_cpg_sites]
results_df.to_csv('cpg_sites_values_with_age.csv')
filtered_cpg_values.to_csv('filtered_cpg_sites_values_with_age0.6.csv', index=True)
filtered_results.to_csv('filtered_cpg_correlation_results.csv', index=False)