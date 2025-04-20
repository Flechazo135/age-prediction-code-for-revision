import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
os.chdir(r"pathway")
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
df=pd.read_excel(r"300MI.xlsx", header=0)
all_mutual_info = df['MI Score'].to_numpy()
selected_indices = np.argsort(all_mutual_info)[-100:]
selected_mutual_info = np.zeros_like(all_mutual_info)
selected_mutual_info[selected_indices] = all_mutual_info[selected_indices]
plt.figure(figsize=(6, 4))
plt.fill_between(range(len(all_mutual_info)), all_mutual_info, color='#3B68B0',alpha=1.0, label='All 4464 Sites')
plt.fill_between(selected_indices, all_mutual_info[selected_indices], color='#D33E1B', alpha=1.0, label='Top 300 Sites')
plt.xlabel('Feature Index',fontweight='bold',color='black',fontsize=12)
plt.ylabel('Mutual Information Score',fontweight='bold',color='black',fontsize=12)
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
