#!/usr/bin/env python
# coding: utf-8

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import pandas as pd
import argparse
import joblib

def train_and_evaluate(df, model_path): 
    # Split data
    # df = pd.read_csv(df_processed_path)
    df = df.sort_values(by='TransactionDT')
    X = df.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train
    clf = lgb.LGBMClassifier(
#         device = 'gpu',
        objective = 'binary',
        verbose=-1,
        metric='auc_roc',
        n_jobs=-1,
        learning_rate = 0.1,
        min_split_gain = 2, 
        n_estimators = 186,
        num_leaves = 36)

    top_150_features = ['V258', 'C1', 'C14', 'V294', 'C13', 'D2', 'card1', 'C7', 'C8', 'V149', 'card2', 'V70', 'D15', 'addr1', 'card6', 'ProductCD', 'TransactionAmt_to_mean_card2', 'C2', 'C11', 'V283', 
                        'P_emaildomain', 'card5', 'C4', 'R_emaildomain', 'V91', 'TransactionAmt_to_std_card3', 'C5', 'V62', 'V308', 'D1', 'V156', 'D3', 'D4', 'C9', 'C6', 'V310', 'D8', 'D10', 'V83', 'V12', 
                        'M4', 'V312', 'TransactionAmt_to_std_card1', 'card3', 'V87', 'M6', 'V45', 'TransactionAmt', 'V80', 'id_20', 'V189', 'V147', 'id_01', 'dist1', 'TransactionAmt_to_std_card2', 'V307', 'M5', 
                        'V314', 'V315', 'V313', 'V165', 'id_02', 'TransactionAmt_to_mean_card4', 'V187', 'V54', 'TransactionAmt_to_std_card5', 'TransactionAmt_to_mean_card6', 'V53', 'id_31', 'V205', 
                        'R_emaildomain_2', 'V201', 'D11', 'V317', 'V61', 'TransactionAmt_to_std_card4', 'id_09', 'id_33', 'id_05', 'V67', 'id_30', 'V158', 'V262', 'TransactionAmt_to_mean_card5', 'V152', 'V38', 
                        'D5', 'V285', 'id_17', 'P_emaildomain_2', 'TransactionAmt_to_std_card6', 'V76', 'V82', 'DeviceInfo', 'V282', 'id_06', 'V291', 'V169', 'V266', 'D12', 'C10', 'V163', 'id_03', 'D6', 'id_32', 
                        'V326', 'V75', 'V261', 'V55', 'DeviceType', 'C12', 'TransactionAmt_to_mean_card3', 'card4', 'D14', 'V128', 'V131', 'V49', 'V99', 'V333', 'V47', 'V78', 'V200', 'M3', 'V30', 'V225', 
                        'TransactionAmt_to_mean_card1', 'D13', 'V256', 'V251', 'V69', 'V56', 'id_04', 'D9', 'V280', 'V279', 'V37', 'V79', 'M2', 'V36', 'V164', 'V48', 'V33', 'id_13', 'V239', 'V150', 'id_37', 'V77',
                        'V126', 'V140', 'V20']
                        
    clf.fit(X_train[top_150_features], y_train)

    # Define metadata
    clf.metadata = {"framework": "LightGBM"}

    # Save the model directly to the specified path
    joblib.dump(clf, model_path) 

    # Evaluate
    y_pred_proba_train = clf.predict_proba(X_train[top_150_features])[:, 1]
    auc_score_train = roc_auc_score(y_train, y_pred_proba_train)
    print(f"AUC-ROC Score (Training Set): {auc_score_train:.4f}")

    y_pred_proba_test = clf.predict_proba(X_test[top_150_features])[:, 1]
    auc_score_test = roc_auc_score(y_test, y_pred_proba_test)
    print(f"AUC-ROC Score (Test Set): {auc_score_test:.4f}")
    
    time_split = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=time_split,
                                scoring='roc_auc', verbose = 1)
    print("Mean cross-validation ROC AUC score (5 splits):", cv_scores.mean())
    
    return {"scores": {"train": auc_score_train, "test": auc_score_test, "cross_validation (5 folds)": cv_scores.mean()}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_processed-path", type=str)
    parser.add_argument("--output-dir", type=str)
    args, _ = parser.parse_known_args()
    _ = train_and_evaluate(args.df_processed_path)
    




