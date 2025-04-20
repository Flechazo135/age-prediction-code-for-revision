import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
os.chdir(r"pathway")
data = pd.read_excel(r"SNaPshot141.xlsx", header=0)
# Other files: pyrosequencing135
train_data = data[data['set'] == 'Training']
test_data = data[data['set'] == 'Test']
X_train = train_data.drop(['age', 'ID', 'set'], axis=1)
y_train = train_data['age']
ID_train = train_data['ID']
X_test = test_data.drop(['age', 'ID', 'set'], axis=1)
y_test = test_data['age']
ID_test = test_data['ID']
# age bin
def age_groups(y, groups):
    bins = np.concatenate(([1], groups, [71]))
    return np.digitize(y, bins) - 1
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
cv_folds = 10
kf = KFold(n_splits=cv_folds, shuffle=True)
classifiers = []
train_predictions = np.zeros(len(y_train))
train_counts = np.zeros(len(y_train))
for fold_num, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    votes_val = np.zeros((X_val.shape[0], 70), dtype=int)
    for i in range(1, 18):
        if i == 1:
            groups = [18,35,52,69]
        else:
            groups = list(range(i, 71,17))
        y_tr_groups = age_groups(y_tr, groups)
        clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
        clf.fit(X_tr, y_tr_groups)
        classifiers.append((clf, groups))
        y_pred_val_groups = clf.predict(X_val)
        for idx, group in enumerate(y_pred_val_groups):
            start = groups[group - 1] if group > 0 else 1
            end = groups[group] if group < len(groups) else 70
            votes_val[idx, start:end] += 1
    y_pred_val = np.argmax(votes_val, axis=1)
    train_predictions[val_index] += y_pred_val
    train_counts[val_index] += 1
final_train_preds = train_predictions / train_counts
mae_train = mean_absolute_error(y_train, final_train_preds)
r2_train = r2_score(y_train, final_train_preds)
adj_r2_train = adjusted_r2(r2_train, len(y_train), X_train.shape[1])
mse_train = mean_squared_error(y_train, final_train_preds)
rmse_train = root_mean_squared_error(y_train, final_train_preds)
print("=== Training Set Metrics ===")
print(f"MAE       : {mae_train:.4f}")
print(f"R²        : {r2_train:.4f}")
print(f"Adj. R²   : {adj_r2_train:.4f}")
print(f"MSE       : {mse_train:.4f}")
print(f"RMSE      : {rmse_train:.4f}")
final_votes_test = np.zeros((X_test.shape[0], 70), dtype=int)
for i in range(1, 18):
    if i == 1:
        groups = [18, 35, 52, 69]
    else:
        groups = list(range(i, 71, 17))
    y_train_groups = age_groups(y_train, groups)
    clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
    clf.fit(X_train, y_train_groups)
    y_test_pred_groups = clf.predict(X_test)
    for idx, group in enumerate(y_test_pred_groups):
        start = groups[group - 1] if group > 0 else 1
        end = groups[group] if group < len(groups) else 70
        final_votes_test[idx, start:end] += 1
final_pred_test = np.argmax(final_votes_test, axis=1)
mae_final = mean_absolute_error(y_test, final_pred_test)
r2_final = r2_score(y_test, final_pred_test)
adj_r2_final = adjusted_r2(r2_final, len(y_test), X_test.shape[1])
mse_final = mean_squared_error(y_test, final_pred_test)
rmse_final = root_mean_squared_error(y_test, final_pred_test)
print("\n=== Final Model on Full Training Set ===")
print(f"MAE       : {mae_final:.4f}")
print(f"R²        : {r2_final:.4f}")
print(f"Adj. R²   : {adj_r2_final:.4f}")
print(f"MSE       : {mse_final:.4f}")
print(f"RMSE      : {rmse_final:.4f}")