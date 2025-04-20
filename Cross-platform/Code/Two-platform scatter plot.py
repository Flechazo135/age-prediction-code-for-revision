import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
os.chdir(r"pathway")
df = pd.read_excel(r"2-Without Dummy results.xlsx", header=0)
X = df['age']
y = df['Predicted_Age']
dummy = df['Dummy']
mae = mean_absolute_error(X, y)
mse = mean_squared_error(X, y)
rmse = np.sqrt(mse)
r2 = r2_score(X, y)
n = len(X)  # number of samples
p = 4       # number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(5,5))
plt.scatter(X[dummy == 0], y[dummy == 0], color='#1B78AB', alpha=1.0, s=35, label='SNaPshot', marker='D')
plt.scatter(X[dummy == 1], y[dummy == 1], color='#e8684a', alpha=1.0, s=35, label='Pyrosequencing', marker='o')
plt.plot([min(X), max(X)], [min(y), max(y)], linestyle='--', color='#000000', linewidth=1.0, alpha=1.0)
plt.xlabel('Chronological Age (years)', fontweight='bold', color='black', fontsize=12)
plt.ylabel('Predicted Age (years)', fontweight='bold', color='black', fontsize=12)
plt.text(0.75, 0.02,
         f"3 CpGs without 'Dummy'\n"
         f"MAE  = {mae:.2f} years\n"
         f"RMSE  = {rmse:.2f} years\n"
         f"MSE = {mse:.2f} years²\n"
         f"R²   = {r2 * 100:.2f}%\n"
         f"Adjusted R² = {adj_r2* 100:.2f}%",
         transform=plt.gca().transAxes,
         color='black',
         fontsize=9,
         verticalalignment='bottom')
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
plt.tight_layout()
plt.show()