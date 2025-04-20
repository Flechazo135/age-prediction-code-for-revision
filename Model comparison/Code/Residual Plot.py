import pandas as pd
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'Times New Roman'
os.chdir(r"pathway")
df = pd.read_excel('SVR_train_regression_results.xlsx')
df['Residual'] = df['Predicted_Age'] - df['True_Age']
# ±1.96 std
residual_std = df['Residual'].std()
upper = 1.96 * residual_std
lower = -1.96 * residual_std
df['Color'] = df['Residual'].apply(lambda x: '#D33E1B' if x > upper or x < lower else '#1B78AB')
edge_color = 'black'
plt.figure(figsize=(8, 5))
for i in range(len(df)):
    plt.scatter(df.loc[i, 'True_Age'], df.loc[i, 'Residual'],
                color=df.loc[i, 'Color'], s=50, alpha=1.0)
plt.axhline(y=0, color='#F8750E', linestyle='--', linewidth=1, label='Zero')
plt.axhline(y=upper, color='gray', linestyle='--', linewidth=1, label='+1.96 SD')
plt.axhline(y=lower, color='gray', linestyle='--', linewidth=1, label='-1.96 SD')
plt.title('Residual Plot (SVR cross-val)', color='black', fontsize=14)
plt.xlabel('Chronological Age (years)', color='black', fontsize=12)
plt.ylabel('Predicted age - Chronological age (years)', color='black', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='upper right',bbox_to_anchor=(1.0, 0.90),fontsize=6)
plt.tight_layout()
plt.show()