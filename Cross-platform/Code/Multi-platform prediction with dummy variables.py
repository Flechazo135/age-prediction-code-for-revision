import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.model_selection import KFold
import os
from sklearn.neural_network import MLPClassifier
os.chdir(r"pathway")
data = pd.read_excel(r"pyrosequencing+SNaPshot+450k.xlsx", header=0)
# Other files: pyrosequencing+SNaPshot.xlsx
X = data.drop(['age', 'ID'], axis=1)
y = data['age']
IDs = data['ID']
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
repeats = 5
cv_folds = 10
predicted_ages = np.zeros((len(y), repeats))
for iteration in range(repeats):
    kf = KFold(n_splits=cv_folds, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        def age_groups(y, groups):
            bins = np.concatenate(([1], groups, [91]))
            return np.digitize(y, bins) - 1
        classifiers = []
        for i in range(1, 18):
            if i == 1:
                groups = [18, 35, 52, 69,86]
            else:
                groups = list(range(i, 91, 17))
            y_train_groups = age_groups(y_train, groups)
            clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000, random_state=2)
            clf.fit(X_train, y_train_groups)
            classifiers.append((clf, groups))

        votes_test = np.zeros((X_test.shape[0], 90), dtype=int)
        for clf, groups in classifiers:
            y_pred_test_groups = clf.predict(X_test)
            for idx, group in enumerate(y_pred_test_groups):
                start = groups[group - 1] if group > 0 else 1
                end = groups[group] if group < len(groups) else 90
                votes_test[idx, start:end] += 1
        y_pred = np.argmax(votes_test, axis=1)

        for idx, pred_age in zip(test_index, y_pred):
            predicted_ages[idx, iteration] = pred_age

average_predicted_ages = np.mean(predicted_ages, axis=1)

output_data = data[['ID'] + list(X.columns) + ['age']].copy()
output_data['Predicted_Age'] = average_predicted_ages
output_file_path_predictions = r"3-With Dummy results.xlsx"
# output_file_path_predictions = r"2-With Dummy results.xlsx"
output_data.to_excel(output_file_path_predictions, index=False)

mae = mean_absolute_error(y, average_predicted_ages)
r2 = r2_score(y, average_predicted_ages)
adj_r2 = adjusted_r2(r2, len(y), X.shape[1])
mse = mean_squared_error(y, average_predicted_ages)
rmse = root_mean_squared_error(y, average_predicted_ages)
print("\n=== Overall Performance Metrics ===")
print(f"MAE       : {mae:.4f}")
print(f"R²        : {r2:.4f}")
print(f"Adj. R²   : {adj_r2:.4f}")
print(f"MSE       : {mse:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"\nPredictions have been saved to: {output_file_path_predictions}")