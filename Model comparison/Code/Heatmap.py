import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
os.chdir(r"pathway")
file_path = 'Training_difference in age groups.xlsx'
data = pd.read_excel(file_path)
data.set_index("Models", inplace=True)
#data_sorted = data.loc[data.mean(axis=1).sort_values().index]
data_sorted = data
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#007bb6','#d0e8f5'])
plt.figure(figsize=(5, 7))
sns.heatmap(data_sorted, annot=True, fmt=".2f", cmap=custom_cmap, annot_kws={"color": "#2C3E50"})
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('')
plt.yticks(rotation=90)
plt.show()