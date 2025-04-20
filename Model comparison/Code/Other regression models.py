import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
os.chdir(r"pathway")
# Input
data = pd.read_excel(r"all sample with set.xlsx", header=0)
train_data = data[data['set'] == 'Training']
test_data = data[data['set'] == 'Testing']
X_train_full = train_data.drop(['age', 'ID', 'set'], axis=1)
y_train_full = train_data['age']
ID_train = train_data['ID']
X_test = test_data.drop(['age', 'ID', 'set'], axis=1)
y_test = test_data['age']
ID_test = test_data['ID']
repeats = 5
cv_folds = 10
predicted_train_ages = np.zeros((len(y_train_full), repeats))
for i in range(repeats):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=i)
    for train_index, val_index in kf.split(X_train_full):
        X_tr, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_tr, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
        model = MLPRegressor(hidden_layer_sizes=(30,15), max_iter=5000,random_state=2)
        # model = LinearRegression()
        # model = KNeighborsRegressor()
        # model = DecisionTreeRegressor(random_state=2)
        # model = SVR()
        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)
        for idx, pred_age in zip(val_index, y_pred_val):
            predicted_train_ages[idx, i] = pred_age
avg_pred_train = np.mean(predicted_train_ages, axis=1)
mae_train = mean_absolute_error(y_train_full, avg_pred_train)
r2_train = r2_score(y_train_full, avg_pred_train)
adj_r2_train = adjusted_r2(r2_train, len(y_train_full), X_train_full.shape[1])
mse_train = mean_squared_error(y_train_full, avg_pred_train)
rmse_train = root_mean_squared_error(y_train_full, avg_pred_train)
print("=== Training Set Metrics (Cross-Validated) ===")
print(f"MAE       : {mae_train:.4f}")
print(f"R²        : {r2_train:.4f}")
print(f"Adj. R²   : {adj_r2_train:.4f}")
print(f"MSE       : {mse_train:.4f}")
print(f"RMSE      : {rmse_train:.4f}")
# save
train_output = pd.DataFrame({
    'ID': ID_train,
    'True_Age': y_train_full,
    'Predicted_Age': avg_pred_train
})
train_output.to_excel("NNR_train_regression_results.xlsx", index=False)
final_model = MLPRegressor(hidden_layer_sizes=(30,15), max_iter=5000,random_state=2)
# final_model = LinearRegression()
# final_model = KNeighborsRegressor()
# final_model = DecisionTreeRegressor(random_state=2)
# final_model = SVR()
final_model.fit(X_train_full, y_train_full)
avg_pred_test = final_model.predict(X_test)
mae_test = mean_absolute_error(y_test, avg_pred_test)
r2_test = r2_score(y_test, avg_pred_test)
adj_r2_test = adjusted_r2(r2_test, len(y_test), X_test.shape[1])
mse_test = mean_squared_error(y_test, avg_pred_test)
rmse_test = root_mean_squared_error(y_test, avg_pred_test)
print("\n=== Test Set Metrics (Final Model) ===")
print(f"MAE       : {mae_test:.4f}")
print(f"R²        : {r2_test:.4f}")
print(f"Adj. R²   : {adj_r2_test:.4f}")
print(f"MSE       : {mse_test:.4f}")
print(f"RMSE      : {rmse_test:.4f}")
test_output = pd.DataFrame({
    'ID': ID_test,
    'True_Age': y_test,
    'Predicted_Age': avg_pred_test
})
test_output.to_excel("NNR_test_regression_results.xlsx", index=False)