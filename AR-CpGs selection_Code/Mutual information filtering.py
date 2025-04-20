import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import os
os.chdir(r"pathway")
df = pd.read_excel("4464.xlsx")
X = df.drop(columns=['ID', 'age'])
y = df['age']
# Calculate mutual information
mi_scores = mutual_info_regression(X, y, discrete_features=False,n_neighbors=55,random_state=123)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
top_300_features = mi_df.sort_values(by='MI Score', ascending=False).head(300)
top_300_cpg_data = df[['ID', 'age'] + list(top_300_features['Feature'])]
top_300_cpg_data.to_excel('top_300_cpg_filtered_with_id_age.xlsx', index=False)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)
mi_df = pd.DataFrame(mi_df).sort_values(by='MI Score', ascending=False)
mi_df.to_excel('300MI.xlsx', index=False)