import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r"pathway")
df = pd.read_excel(r"Line charts_Testing_R2.xlsx", header=0)
bins = df['Bins']
NN  = df['NN']
LDA = df['LDA']
SVM= df['SVM']
KNN = df['KNN']
DT = df['DT']
LR= df['LR']
NB = df['NB']
plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(1.5)
# Each classifier uses different colors and symbols
plt.plot(bins, NN, marker='o', color='#7150B4', label='NN', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, LDA, marker='o', color='#EC7D0E', label='LDA', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, SVM, marker='o', color='#359787', label='SVM', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, KNN, marker='o', color='#C63D18', label='KNN', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, DT, marker='o', color='#3C78BA', label='DT', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, LR, marker='o', color='#2A87AC', label='LR', markersize=3, linewidth=1.5, alpha=1.0)
plt.plot(bins, NB, marker='o', color='#E69F00', label='NB', markersize=3, linewidth=1.5, alpha=1.0)
bar_colors = ['#7150B4', '#3C78BA', '#2A87AC', '#359787', '#E69F00', '#EC7D0E', '#C63D18']
ax.set_xlabel('Bins (years)', fontsize=12, color='black', fontweight='bold')
ax.set_ylabel('R²', fontsize=12, color='black', fontweight='bold')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.legend(loc='upper left', bbox_to_anchor=(0.77, 0.35))
# For MAE, RMSE, MSE; bbox_to_anchor=(0.77, 1.05)
plt.show()