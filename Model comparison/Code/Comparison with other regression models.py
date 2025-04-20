import pandas as pd
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['font.family'] = 'Times New Roman'
os.chdir(r"pathway")
file_path = 'Training_Comparison with other regression models.xlsx'
data_df = pd.read_excel(file_path)
# Calculate the mean and standard error of the mean (SEM)
grouped_data = data_df.groupby('Models')['Absolute error'].agg(['mean', 'sem']).reset_index()
grouped_data = grouped_data.sort_values(by='mean', ascending=True)
categories = grouped_data['Models'].tolist()
means = grouped_data['mean'].tolist()
std_errors = grouped_data['sem'].tolist()
bar_colors = ['#7150B4', '#3C78BA',  '#359787', '#E69F00', '#EC7D0E', '#C63D18']
error_colors = ['#6748A6', '#267C9E', '#2F8577', '#DA9700', '#D9730D', '#B93A17']
fig, ax = plt.subplots(figsize=(8, 6))
bars = []
for i, (category, mean, std_err, bar_color, err_color) in enumerate(zip(categories, means, std_errors, bar_colors, error_colors)):
    bar = ax.bar(category, mean, yerr=std_err, capsize=10, color=bar_color, alpha=0.7, edgecolor=bar_color,
                  error_kw={'ecolor': err_color, 'elinewidth': 1.8})
    bars.append(bar)
ax.scatter(categories, means, color='red', alpha=0.7, marker='o', s=75, label='Mean')
ax.set_ylabel('Absolute error (years)', color='black', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()