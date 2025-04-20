import pandas as pd
import os
os.chdir(r"pathway")
# Input
df = pd.read_excel('all samples.xlsx')
def sample_by_age(df):
    sampled = []
    for age, group in df.groupby('age'):
        if len(group) >= 2:
            sampled.append(group.sample(n=1, random_state=5))
    return pd.concat(sampled)
filtered_df = sample_by_age(df)
filtered_df.to_excel('Samples-split.xlsx', index=False)