import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
os.chdir(r"pathway")
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
df=pd.read_csv(r"cpg_sites_values_with_age.csv", header=0)
df['-log10(Spearman_p-value)'] = -np.log10(df['Spearman_p-value'])
def assign_color(correlation):
    if correlation >= 0.6:
        return '#D33E1B'
    elif correlation <= -0.6:
        return '#D33E1B'
    else:
        return '#5d7092'
df['Color'] = df['Spearman_Correlation'].apply(assign_color)
plt.figure(figsize=(6,6))
sns.scatterplot(
    x='Spearman_Correlation',
    y='-log10(Spearman_p-value)',
    data=df,
    c=df['Color'],
    legend=False
)
plt.axvline(x=0.6, color='black', linestyle='--', linewidth=1)
plt.axvline(x=-0.6, color='black', linestyle='--', linewidth=1)
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1)
plt.xlabel('Spearman Correlation',fontweight='bold',color='black',fontsize=12)
plt.ylabel('-log10(Spearman p-value)',fontweight='bold',color='black',fontsize=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.legend()
plt.show()