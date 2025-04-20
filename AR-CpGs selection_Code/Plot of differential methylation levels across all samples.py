import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(r"pathway")
df = pd.read_excel(r"Manually Processed STEP1_4464.xlsx", header=0)
df['ID'] = range(1, len(df) + 1)
colors = ['#D33E1B' if x > 0.4 else '#3B68B0' for x in df['diff']]
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6,6))
plt.scatter(df['ID'], df['diff'], color=colors,s=12, alpha=0.8)
plt.axhline(y=0.4, color='red',alpha=0.8, linestyle='--',linewidth=1.0)
for i, diff_value in enumerate(df['diff']):
    if diff_value == 0.400011944:
        plt.axvline(x=df['ID'][i], color='red',alpha=0.8, linestyle='--',linewidth=1.0)
plt.xlabel('ID',fontweight='bold',color='black',fontsize=12)
plt.ylabel('Methylation Variation Range',fontweight='bold',color='black',fontsize=12)
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