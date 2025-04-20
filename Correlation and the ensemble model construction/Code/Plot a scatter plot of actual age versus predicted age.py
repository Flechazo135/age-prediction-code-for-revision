import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
# Input
os.chdir(r"pathway")
df = pd.read_excel(r"283_test_results.xlsx", header=0)
X = df['True_Age']
y = df['Predicted_Age']
n = len(X)
p = 10  # 10CpGs
r2 = r2_score(X, y)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(X, y)
mse = mean_squared_error(X, y)
rmse = np.sqrt(mse)
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='#1B78AB', alpha=0.9, s=70)
plt.plot([min(X), max(X)], [min(X), max(X)], linestyle='--', color='#F8750E', linewidth=1.0)
plt.xlabel('Chronological Age (years)', fontweight='bold', color='black', fontsize=12)
plt.ylabel('Predicted Age (years)', fontweight='bold', color='black', fontsize=12)
metrics_text = (f"MAE = {mae:.2f} years\n"
                f"RMSE = {rmse:.2f} years\n"
                f"MSE = {mse:.2f} years²\n"
                f"R² = {r2 * 100:.2f}%\n"
                f"Adjusted R² = {adjusted_r2 * 100:.2f}%")
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', color='black')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.tight_layout()
plt.show()