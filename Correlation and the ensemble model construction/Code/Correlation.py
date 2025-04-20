import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
os.chdir(r"pathway")
df = pd.read_excel(r"all samples.xlsx", header=0)
columns = ['cg07547549', 'cg00481951', 'cg21296230', 'cg10501210', 'cg13654588',
           'cg14361627', 'cg17885226', 'cg17110586', 'cg15480367', 'cg19671120']
y = 'age'
x_min = df[columns].min().min()
x_max = df[columns].max().max()
y_min = df[y].min() - 7
y_max = df[y].max() + 9
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
colors = ['#1C7FB6', '#1C7FB6', '#1C7FB6', '#1C7FB6', '#1C7FB6',
          '#E76C07', '#E76C07', '#E76C07', '#E76C07', '#E76C07']
fig, axs = plt.subplots(2, 5, figsize=(18, 6))  # 2-row 5-column subplot grid
for i, col in enumerate(columns):
    ax = axs[i // 5, i % 5]
    # Swap x and y for regplot
    sns.regplot(x=df[y], y=df[col], data=df, ax=ax, color=colors[i], scatter_kws={'s': 10, 'alpha': 0.8},
                line_kws={'linewidth': 1.5, 'alpha': 0.7}, label=None)
    # Calculate Spearman correlation coefficient
    corr, p_value = spearmanr(df[col], df[y])
    label = f'{col} \n(r={corr:.2f} p={p_value:.2e})'
    # Set axis limits
    x_min = df[y].min() - 4
    x_max = df[y].max() + 4
    y_min = df[col].min() - 0.02
    y_max = df[col].max() + 0.02
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Age (years)', fontsize=10, color='black', fontweight='bold')
    ax.set_ylabel('DNAm', fontsize=10, color='black', fontweight='bold')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    if i == 0:
        ax.text(0.97, 0.95, label, ha='right', va='top', transform=ax.transAxes, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    else:
        ax.text(0.965, 0.95, label, ha='right', va='top', transform=ax.transAxes, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()