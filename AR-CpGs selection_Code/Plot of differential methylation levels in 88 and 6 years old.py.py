import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
os.chdir(r"pathway")
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
df = pd.read_excel(r"all_cpg_diff_sorted.xlsx", header=0)
total_mae = df['Difference'].to_numpy()
ranking = np.argsort(np.argsort(-total_mae))  # Rank by descending order of the difference
selected_indices = np.argsort(total_mae)[-100:]
selected_mae = np.zeros_like(total_mae)
selected_mae[selected_indices] = total_mae[selected_indices]
smoothed_mae = savgol_filter(total_mae, window_length=100, polyorder=90)
plt.figure(figsize=(6, 4))
plt.fill_between(total_mae, ranking, color='#507CC4', alpha=1.0, label='All 300 Sites')
plt.fill_between(total_mae[selected_indices], ranking[selected_indices], color='#D33E1B', alpha=1.0, label='Top 100 Sites')
plt.xlabel('Difference', fontweight='bold', color='black', fontsize=12)
plt.ylabel('Ranking', fontweight='bold', color='black', fontsize=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.legend()
plt.show()