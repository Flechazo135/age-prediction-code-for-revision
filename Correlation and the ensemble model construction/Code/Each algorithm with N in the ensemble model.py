import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
import os
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
# age boin
def age_groups(y, groups):
    bins = np.concatenate(([1], groups, [70]))
    return np.digitize(y, bins) - 1
# adjusted_r2
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
rkf = RepeatedKFold(n_splits=10, n_repeats=5)
all_results = []
# N = 1-53
for N in range(1, 54):
    print(f"\n--- Evaluating N = {N} ---")
    predicted_train = []
    actual_train = []
    classifiers_groups = []
    for i in range(1, N + 1):
        if i == 1:
            groups = list(range(N + 1, 70, N))
        else:
            groups = list(range(i, 71, N))
        classifiers_groups .append(groups)
    # training with 5 × 10 cross-validation
    for train_idx, val_idx in rkf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        models = []
        for groups in classifiers_groups:
            y_tr_groups = age_groups(y_tr, groups)
            clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
            # clf = DecisionTreeClassifier(random_state=2) *DT*
            # clf = KNeighborsClassifier() *KNN*
            # clf = LinearDiscriminantAnalysis() *LDA*
            # clf = LogisticRegression(random_state=2) *LR*
            # clf = GaussianNB() *NB*
            # clf = SVC(probability=True,random_state=2) *SVM*
            clf.fit(X_tr, y_tr_groups)
            models.append((clf, groups))
        for i in range(X_val.shape[0]):
            votes = np.zeros(70, dtype=int)
            for clf, groups in models:
                pred_group = clf.predict(X_val.iloc[[i]])
                bins = np.concatenate(([1], groups, [70]))
                group_idx = pred_group[0]
                vote_range = range(bins[group_idx], bins[group_idx + 1])
                for age in vote_range:
                    votes[age - 1] += 1
            pred_age = np.argmax(votes) + 1
            predicted_train.append(pred_age)
            actual_train.append(y_val.values[i])
    train_mae = mean_absolute_error(actual_train, predicted_train)
    train_r2 = r2_score(actual_train, predicted_train)
    train_rmse = root_mean_squared_error(actual_train, predicted_train)
    train_mse = mean_squared_error(actual_train, predicted_train)
    train_adj_r2 = adjusted_r2(train_r2, len(actual_train), X_train.shape[1])
    # all training
    final_models = []
    for groups in classifiers_groups:
        y_train_groups = age_groups(y_train, groups)
        clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=5000,random_state=2)
        # clf = DecisionTreeClassifier(random_state=2) *DT*
        # clf = KNeighborsClassifier() *KNN*
        # clf = LinearDiscriminantAnalysis() *LDA*
        # clf = LogisticRegression(random_state=2) *LR*
        # clf = GaussianNB() *NB*
        # clf = SVC(probability=True,random_state=2) *SVM*
        clf.fit(X_train, y_train_groups)
        final_models.append((clf, groups))
    predicted_test = []
    for i in range(X_test.shape[0]):
        votes = np.zeros(70, dtype=int)
        for clf, groups in final_models:
            pred_group = clf.predict(X_test.iloc[[i]])
            bins = np.concatenate(([1], groups, [70]))
            group_idx = pred_group[0]
            vote_range = range(bins[group_idx], bins[group_idx + 1])
            for age in vote_range:
                votes[age - 1] += 1
        pred_age = np.argmax(votes) + 1
        predicted_test.append(pred_age)
    # testing
    test_mae = mean_absolute_error(y_test, predicted_test)
    test_r2 = r2_score(y_test, predicted_test)
    test_rmse = root_mean_squared_error(y_test, predicted_test)
    test_mse = mean_squared_error(y_test, predicted_test)
    test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test.shape[1])
    all_results.append({
        'N': N,
        'Train MAE': train_mae,
        'Train R2': train_r2,
        'Train Adjusted R2': train_adj_r2,
        'Train RMSE': train_rmse,
        'Train MSE': train_mse,
        'Test MAE': test_mae,
        'Test R2': test_r2,
        'Test Adjusted R2': test_adj_r2,
        'Test RMSE': test_rmse,
        'Test MSE': test_mse
    })
    print(f"N={N} | Train MAE={train_mae:.4f} | Test MAE={test_mae:.4f}")
# save
results_df = pd.DataFrame(all_results)
results_df.to_excel("Ensemble-N-NN.xlsx", index=False)
