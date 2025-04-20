import pandas as pd
import os
os.chdir(r"pathway")
df = pd.read_excel("top_300_cpg_filtered_with_id_age.xlsx")
# Obtain sample data for 6-year-old and 88-year-old subjects
age_6_samples = df[df['age'] == 6].iloc[:, 2:].mean()
age_88_samples = df[df['age'] == 88].iloc[:, 2:].mean()
# Calculate the absolute difference for each CpG site
diff = (age_88_samples - age_6_samples).abs()
# Select the top 100 CpG sites with the largest absolute differences
top_100_cpgs = diff.nlargest(100)
top_100_cpgs_data = df[['ID', 'age'] + top_100_cpgs.index.tolist()]
output_file_path = 'top_100_cpg_diff.xlsx'
top_100_cpgs_data.to_excel(output_file_path, index=False)
all_cpg_diff = pd.DataFrame({
    'CpG_site': diff.index,
    'Difference': diff.values
}).sort_values(by='Difference', ascending=False)
output_all_diff_file_path = 'all_cpg_diff_sorted.xlsx'
all_cpg_diff.to_excel(output_all_diff_file_path, index=False)