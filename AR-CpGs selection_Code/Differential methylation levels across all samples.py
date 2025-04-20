import pandas as pd
import os
os.chdir(r"pathway")
df = pd.read_csv("filtered_cpg_sites_values_with_age0.6.csv")
df = df.drop(columns=["ID"])
max_values_per_column = df.max(axis=0)
min_values_per_column = df.min(axis=0)
# Calculate the difference between the maximum and minimum values
diff_values_per_column = max_values_per_column - min_values_per_column
diff_df = pd.DataFrame({'max': max_values_per_column,
                        'min': min_values_per_column,
                        'diff': diff_values_per_column})
diff_df.to_excel('STEP1：4464.xlsx', sheet_name='diff')
filtered_columns = df.loc[:, diff_values_per_column > 0.4]
with pd.ExcelWriter('STEP1：4464.xlsx', mode='a') as writer:
    filtered_columns.to_excel(writer, sheet_name='diff above 0.4')
filtered_columns.to_excel('4464.xlsx')