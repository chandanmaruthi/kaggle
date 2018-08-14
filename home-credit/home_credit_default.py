import matplotlib
import numpy as np
#1import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
from pandas_summary import DataFrameSummary
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import lightgbm as lgb

models = []




def handle_categoricals(df, all_data):
    # Create a label encoder object
    le = LabelEncoder()

    le_count = 0
    oh_count = 0
    oh_df = pd.DataFrame()
    # Iterate through the columns
    for col in df:
        if col == 'TARGET':
            continue
        all_df = pd.DataFrame()
        all_df[col] = all_data[col]
        print(all_df.describe)
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                print('le :' + col)
                # Train on the training data
                le.fit(all_df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                # df_application[col] = le.transform(df_application[col])

                # Keep track of how many columns were label encoded
                le_count += 1
            else:
                oh_df = pd.concat([oh_df,pd.get_dummies(df[col].astype(str),prefix=col+'_')],axis=1)
                # print('ce: ' + col)
                # le = LabelEncoder()
                # ohe = OneHotEncoder(sparse=False)
                # df_tmp = pd.DataFrame()
                # le.fit(all_df[col].astype(str))
                # df_tmp[col] = le.transform(all_df[col].astype(str))
                # df_tmp[col]= df_tmp[col].reshape(-1,1)
                # print(df_tmp[col].shape)
                # ohe.fit(df_tmp[col].reshape(len(df_tmp[col]),1))
                # f = pd.DataFrame()
                # f[col] = le.transform(df[col].astype(str))
                # g = pd.DataFrame()
                # g = ohe.transform(f[col].reshape(len(f[col]),1))
                # oh_df = pd.concat([oh_df, pd.DataFrame(data=g)], axis=1)
                df.drop([col], axis=1, inplace=True)

    print('%d columns were label encoded.' % le_count)

    df= pd.concat([df,oh_df],axis=1)
    print(df.columns)
    return df

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
    return d
def handle_nulls(df):
    df.dropna(axis=1,inplace=True)
    return df
def compute_score(clf, X, y, scoring='accuracy'):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    xval = cross_val_score(clf, X, y, cv = 10, scoring=scoring)
    return np.mean(xval)

def gen_train_test_split(df):
    from sklearn.model_selection import train_test_split
    X = df.drop(['TARGET'],axis=1)
    y = df['TARGET']
    train_features, test_features, train_labels, test_labels = \
    train_test_split(X, y, test_size=0.1, random_state=42)
    return train_features, test_features, train_labels, test_labels

def model_rf(X, y):
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators = 100,
                                      random_state = 42,
                                      n_jobs=-1,
                                      oob_score=True,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features='auto',
                                      bootstrap=True)
    # Train the model on training data
    rf_model.fit(X, y)
    return  rf_model



def model_lgbm(X,y,X_test=None,y_test=None):
    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    train_data = lgb.Dataset(X, label=y)
    bst = lgb.train(param, train_data, num_round)
    num_round = 10
    lgb.cv(param, train_data, num_round, nfold=5)
    return bst

def score_model(model, X_test, y_test):
    #predictions = rf_model.predict(test_features)
    #accuracy =  compute_score(rf_model,test_features,test_labels)
    #print(accuracy)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    print(auc(false_positive_rate, true_positive_rate))
    score = roc_auc_score(y_test, model.predict(X_test))
    return score

def record_model(model_name, model, score):
    model_inst = {}
    model_inst['model'] = model
    model_inst['model_name'] = model_name
    model_inst['score'] = score
    print('{}: {}'.format(model_name, score))
    models.append(model_inst)

def run(df, df_test, df_all, train=True):
    print('1')
    df = handle_categoricals(df, df_all)
    print('2')
    df = handle_nulls(df)
    df_test = handle_categoricals(df_test,df_all)

    print(df.shape)
    print(df_test.shape)
    df = add_missing_dummy_columns(df, df_test.columns)
    df_test = add_missing_dummy_columns(df_test, df.columns)
    print(df.shape)
    print(df.shape)
    df_test = df_test.fillna(method='ffill')
    #exit()
    X_train, X_test, y_train, y_test = gen_train_test_split(df)
    model = model_rf(X_train, y_train)
    score = score_model(model, X_test, y_test)
    record_model('random_forest',model, score)
    #model = model_lgbm(X_train, y_train)
    y_pred = model.predict(X_test)
    score = roc_auc_score(y_test, model.predict(X_test))

    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict(X_test))
    #score = score_model(model, X_test, y_test)
    #record_model('lgbm',model, score)
    print('4')
    score = 0

    selected_model = None

    for model_inst in models:
        if model_inst['score'] > score:
            score = model_inst['score']
            selected_model = model_inst['model']
    prediction = selected_model.predict(df_test)
    submission_df = pd.DataFrame()
    submission_df['SK_ID_CURR'] = df_test['SK_ID_CURR']
    submission_df['TARGET'] = prediction
    submission_df.to_csv('home_credit_submission.csv',',', index=False)

#test
# bureau_balance.csv
# bureau.csv
# credit_card_balance.csv
# installments_payments.csv
# POS_CASH_balance.csv
# previous_application.csv
# sample_submission.csv

df_application = pd.read_csv('application_train.csv')
df_application_test = pd.read_csv('application_test.csv')
df_application = df_application[:1000]
df_application_test = df_application_test[:1000]

# le = LabelEncoder()
# le.fit(df_application['NAME_TYPE_SUITE'].astype(str))
# df_tmp = le.transform(df_application['NAME_TYPE_SUITE'].astype(str))
# ohe=OneHotEncoder()
# ohe.fit(df_tmp)

all_data = pd.concat([df_application, df_application_test],axis=0)
run(df_application,df_application_test,all_data, True)
#run(df_application, df_application_test, all_data, False)