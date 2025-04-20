import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import lime.lime_tabular
os.chdir(r"pathway")
# Input
data = pd.read_excel(r"all sample with set.xlsx", header=0)
train_data = data[data['set'] == 'Training']
test_data = data[data['set'] == 'Testing']
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
repeats = 5
cv_folds = 10
classifiers = []
train_mae_scores = []
train_preds = [[] for _ in range(len(y_train))]
for repeat in range(repeats):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=repeat)
    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        votes_te = np.zeros((X_val.shape[0], 70), dtype=int)
        for i in range(1, 18):
            if i == 1:
                groups = [18, 35, 52, 69]
            else:
                groups = list(range(i, 71, 17))
            y_tr_groups = age_groups(y_tr, groups)
            clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
            clf.fit(X_tr, y_tr_groups)
            classifiers.append((clf, groups))
            y_pred_val_groups = clf.predict(X_val)
            for idx, group in enumerate(y_pred_val_groups):
                start = groups[group - 1] if group > 0 else 1
                end = groups[group] if group < len(groups) else 70
                votes_te[idx, start:end] += 1
        y_pred_val = np.argmax(votes_te, axis=1)
        for idx, pred in zip(val_index, y_pred_val):
            train_preds[idx].append(pred)
        mae_te = mean_absolute_error(y_val, y_pred_val)
        train_mae_scores.append(mae_te)

avg_train_mae = np.mean(train_mae_scores)
print(f"Average Training MAE (per fold): {avg_train_mae:.4f}")
y_pred_train_final = np.array([int(np.round(np.mean(preds))) for preds in train_preds])
mae_train_final = mean_absolute_error(y_train, y_pred_train_final)
r2_train_final = r2_score(y_train, y_pred_train_final)
print(f"Final Training MAE (averaged predictions): {mae_train_final:.4f}")
print(f"Final Training R² (averaged predictions): {r2_train_final:.4f}")
train_output_data = pd.DataFrame({
    'ID': ID_train,
    'True_Age': y_train,
    'Predicted_Age': y_pred_train_final
})
train_output_file_path = r"283_train_results.xlsx"
train_output_data.to_excel(train_output_file_path, index=False)
print(f"Training predictions have been saved to: {train_output_file_path}")
final_classifiers = []
votes_test_final = np.zeros((X_test.shape[0], 70), dtype=int)
for i in range(1, 18):
    if i == 1:
        groups = [18, 35, 52, 69]
    else:
        groups = list(range(i, 71, 17))
    y_train_groups = age_groups(y_train, groups)
    clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
    clf.fit(X_train, y_train_groups)
    final_classifiers.append((clf, groups))
    # testing
    y_pred_test_groups = clf.predict(X_test)
    for idx, group in enumerate(y_pred_test_groups):
        start = groups[group - 1] if group > 0 else 1
        end = groups[group] if group < len(groups) else 70
        votes_test_final[idx, start:end] += 1
y_pred_test_final = np.argmax(votes_test_final, axis=1)
mae_test_final = mean_absolute_error(y_test, y_pred_test_final)
r2_test_final = r2_score(y_test, y_pred_test_final)
print(f"\nFinal Model (trained on all training data):")
print(f"Test MAE: {mae_test_final:.4f}")
print(f"Test R²: {r2_test_final:.4f}")
final_output_data = pd.DataFrame({
    'ID': ID_test,
    'True_Age': y_test,
    'Predicted_Age': y_pred_test_final
})
final_output_file_path = r"283_test_results.xlsx"
final_output_data.to_excel(final_output_file_path, index=False)
print(f"Final test predictions have been saved to: {final_output_file_path}")
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
def adjusted_r2_score(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
n_train = len(y_train)
p_train = X_train.shape[1]
mae_train = mean_absolute_error(y_train, y_pred_train_final)
r2_train = r2_score(y_train, y_pred_train_final)
mse_train = mean_squared_error(y_train, y_pred_train_final)
rmse_train = np.sqrt(mse_train)
adj_r2_train = adjusted_r2_score(y_train, y_pred_train_final, n_train, p_train)
print(f"Training MAE: {mae_train:.4f}")
print(f"Training R²: {r2_train:.4f}")
print(f"Training Adjusted R²: {adj_r2_train:.4f}")
print(f"Training MSE: {mse_train:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
n_test = len(y_test)
p_test = X_test.shape[1]
mae_test = mean_absolute_error(y_test, y_pred_test_final)
r2_test = r2_score(y_test, y_pred_test_final)
mse_test = mean_squared_error(y_test, y_pred_test_final)
rmse_test = np.sqrt(mse_test)
adj_r2_test = adjusted_r2_score(y_test, y_pred_test_final, n_test, p_test)
print(f"\nTest MAE: {mae_test:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Test Adjusted R²: {adj_r2_test:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")


import lime.lime_tabular
import matplotlib.pyplot as plt
import os

def vote_predict_proba(X_df):
    if isinstance(X_df, np.ndarray):
        X_df = pd.DataFrame(X_df, columns=X_train.columns)

    votes = np.zeros((X_df.shape[0], 70), dtype=float)
    for clf, groups in classifiers:
        y_pred_groups = clf.predict(X_df)
        for idx, group in enumerate(y_pred_groups):
            start = groups[group - 1] if group > 0 else 1
            end = groups[group] if group < len(groups) else 70
            votes[idx, start:end] += 1

    probs = votes + 1e-3
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=[str(i) for i in range(70)],
    mode='classification'
)
lime_csv_dir = "lime_csv"
lime_img_dir = "lime_img"
os.makedirs(lime_csv_dir, exist_ok=True)
os.makedirs(lime_img_dir, exist_ok=True)
all_lime = []
for idx in range(len(X_test)):
    sample = X_test.iloc[idx]
    exp = explainer_lime.explain_instance(
        data_row=sample.values,
        predict_fn=vote_predict_proba,
        num_features=10
    )
    # save csv
    exp_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
    exp_df['Sample_Index'] = idx
    exp_df.to_csv(f"{lime_csv_dir}/lime_sample_{idx}.csv", index=False)
    # save fig
    fig = exp.as_pyplot_figure()
    fig.savefig(f"{lime_img_dir}/lime_sample_{idx}.png", dpi=300)
    plt.close(fig)
    all_lime.append(exp_df)
# all save
all_lime_df = pd.concat(all_lime, ignore_index=True)
all_lime_df.to_csv("all_lime_explanations.csv", index=False)
