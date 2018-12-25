import matplotlib
import numpy as np
import os
import sys
#1import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
from pandas_summary import DataFrameSummary
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

import multiprocessing as mp
from collections import Counter
import psutil
warnings.filterwarnings("ignore")

models = []
parallel_funcs = []
parallel_funcs_2 =[]

#class util:

def handle_categoricals(df, df_test, train=True):
    # Create a label encoder object
    all_df_data = pd.concat([df,df_test],axis=0)

    le_count = 0
    oh_count = 0
    oh_df = pd.DataFrame()
    # Iterate through the columns
    for col in df:
        le = LabelEncoder()
        if col == 'TARGET':
            continue
        print(col[:2])
        if col[:2]=="te_":
            continue
        all_df = pd.DataFrame()
        all_df[col] = all_df_data[col]

        if str(df[col].dtype) in ['object', 'category']:
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                le.fit(all_df[col].astype(str))
                #df['le_'+str(col)] = le.transform(df[col].astype(str))
                le_count += 1
            else:
                oh_count +=1
                oh_df = pd.concat([oh_df,pd.get_dummies(df[col].astype(str),prefix=col+'_')],axis=1)



    print('%d columns were label encoded.' % le_count)
    print('%d columns were one hot encoded.' % oh_count)

    df= pd.concat([df,oh_df],axis=1)
    return df

def handle_categoricals_target_encode(df_train, df_test):
    for col in df_train:
        if col == 'TARGET':
            continue
        if str(df_train[col].dtype) in ['object', 'category']:
            #Target Encoding
            print("target encoding {}".format(col))
            te = ce.TargetEncoder(impute_missing=False, handle_unknown='ignore')
            new_cats = list(set(df_train[col].unique()) -set(df_test[col].unique()))
            new_df = df_train[[col,"TARGET"]]
            for new_cat in new_cats:
                new_df.append([new_cat,0])
            te.fit(new_df[col].values,new_df['TARGET'].values)
            df_train['te_'+str(col)] =  te.transform(df_train[col].values)
            df_test['te_' + str(col)] = te.transform(df_test[col].values)
            # df_train.drop([col],axis=1, inplace=True)
            # df_test.drop([col], axis=1, inplace=True)
    return df_train, df_test

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        if c != 'TARGET':
            d[c] = 99999
    return d

def handle_nulls(df):
    for col in df:
        if col == 'TARGET':
            continue
        #print(df[col].dtype)
        try:
            if str(df[col].dtype) == 'object':
                #df[col].add_categories(['un_specified'])
                df[col] = df[col].fillna(999999)
            elif str(df[col].dtype) == 'category':
                #print(str(df[col].dtype))
                df[col] = df[col].cat.add_categories([99999]).fillna(99999)
            else:
                df[col] = df[col].fillna(99999)
        except:

            df[col] = df[col].fillna(99999)
    return df

def gen_train_test_split(df):
    from sklearn.model_selection import train_test_split
    X = df.drop(['TARGET'],axis=1)
    y = df['TARGET']
    train_features, test_features, train_labels, test_labels = \
    train_test_split(X, y, test_size=0.1, random_state=42)
    return train_features, test_features, train_labels, test_labels
# scale values
def scale_values(df, col_name):
    new_col_name = 'scaled_' + str(col_name)
    df[new_col_name] = df[col_name] - df[col_name].min()
    df[new_col_name] = df[col_name] / df[col_name].max()
    return df[new_col_name]

def log_transform_values( df, col_name):
    new_col_name = 'log_transformed_' + str(col_name)
    df[new_col_name] = df[col_name] - df[col_name].min()
    df[new_col_name] = df[col_name] - df[col_name].max()
    df[new_col_name].apply(np.log)
    return df[new_col_name]

def min_max_scale_values(df, col_name):
    from sklearn.preprocessing import MinMaxScaler
    new_col_name = 'min_max_scaled' + str(col_name)
    scaler = MinMaxScaler()
    df[new_col_name] = pd.DataFrame(scaler.fit_transform(df), columns=[new_col_name])
    return df[new_col_name]

def standard_scale_values(df, col_name):
    import sklearn.preprocessing as preproc
    new_col_name = 'standard_scaled' + str(col_name)
    df[new_col_name] = preproc.StandardScaler().fit_transform(df[[col_name]])
    return df[new_col_name]

def binn_col(df,col,buckets,labels):
    return pd.cut(df[col],buckets, labels=labels)


def bin_col_q_bins(df,col, bins=10):
    return pd.qcut(df[col],bins, labels=False,duplicates='drop')

def make_subset(df, subset=0):
    if subset >0:
        df = df[:subset]
    return df

def load_intermediate():
    df_application = pd.read_pickle('../in_process/df_train.pickle')
    df_application_test = pd.read_pickle('../in_process/df_test.pickle')
    return df_application, df_application_test

def save_intermediate(df_application, df_application_test):
    df_application.to_pickle('./data/df_train.pickle')
    df_application_test.to_pickle('./data/df_test.pickle')
