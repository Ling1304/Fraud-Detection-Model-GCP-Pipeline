#!/usr/bin/env python
# coding: utf-8

# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_cleaning(): 
    # Read the CSV file into a DataFrame
    train_id_path = 'gs://norse-voice-440615-v3_cloudbuild/source/data/train_identity.csv'
    train_transaction_path = 'gs://norse-voice-440615-v3_cloudbuild/source/data/train_transaction.csv'

    df_trans = pd.read_csv(train_transaction_path)
    df_id = pd.read_csv(train_id_path)
    
    # Combine train_transaction and train_identity together
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    # Data Cleaning
    null_cols = [col for col in df.columns if (df[col].isnull().sum() / df.shape[0]) > 0.9]
    freq_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    cols_to_drop = list(set(null_cols + freq_cols + one_value_cols))
    cols_to_drop.remove('isFraud')
    df = df.drop(cols_to_drop, axis=1)
    return df

def data_preprocessing(df):
    # Feature Engineering
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card2'] = df['TransactionAmt'] / df.groupby(['card2'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card3'] = df['TransactionAmt'] / df.groupby(['card3'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card5'] = df['TransactionAmt'] / df.groupby(['card5'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card6'] = df['TransactionAmt'] / df.groupby(['card6'])['TransactionAmt'].transform('mean')

    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card2'] = df['TransactionAmt'] / df.groupby(['card2'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card3'] = df['TransactionAmt'] / df.groupby(['card3'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card5'] = df['TransactionAmt'] / df.groupby(['card5'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card6'] = df['TransactionAmt'] / df.groupby(['card6'])['TransactionAmt'].transform('std')
    
    df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df['P_emaildomain'].str.split('.', expand=True)
    df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df['R_emaildomain'].str.split('.', expand=True)
    df = df.drop(['P_emaildomain_3', 'R_emaildomain_3'], axis=1) 
    
    # Label Encoding
    cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(list(df[col].astype(str).values))
            df[col] = le.transform(list(df[col].astype(str).values))
    
    return df



