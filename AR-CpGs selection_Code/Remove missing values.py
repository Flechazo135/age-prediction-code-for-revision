import pandas as pd
import os
os.chdir(r"pathway")
df=pd.read_excel("GSE59509.xlsx",header=0)
#Other files: GSE78874.xlsx, GSE92767.xlsx, GSE111223.xlsx, GSE119078.xlsx, GSE138279.xlsx
rows_with_na = df[df.isna().any(axis=1)].index.tolist()
df_cleaned = df.dropna(axis=0)
df_cleaned.to_csv('GSE59509_removing missing.csv', index=False)