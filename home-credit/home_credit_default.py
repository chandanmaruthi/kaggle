import matplotlib
import numpy as np
#1import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
df_application = pd.read_csv('application_train.csv')
df_application_test = pd.read_csv('application_test.csv')
from pandas_summary import DataFrameSummary
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
#test
# bureau_balance.csv
# bureau.csv
# credit_card_balance.csv
# installments_payments.csv
# POS_CASH_balance.csv
# previous_application.csv
# sample_submission.csv

df_application = df_application
df_application_test = df_application_test

df_application.head(5)
df_application.shape


models = []

def handle_categoricals(df):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    oh_count = 0
    oh_df = pd.DataFrame()
    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                print('le :' + col)
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                # df_application[col] = le.transform(df_application[col])

                # Keep track of how many columns were label encoded
                le_count += 1
            else:
                print('ce: ' + col)
                oh_df = pd.concat([oh_df, pd.get_dummies(df[col], prefix=col + '_')], axis=1)
                df.drop([col], axis=1, inplace=True)

    print('%d columns were label encoded.' % le_count)

    df= pd.concat([df,oh_df],axis=1)
    return df

def drop_nulls(df):
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
def score_model(model, X_text, y_test):
    #predictions = rf_model.predict(test_features)
    #accuracy =  compute_score(rf_model,test_features,test_labels)
    #print(accuracy)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(X_text)[:,1])
    print(auc(false_positive_rate, true_positive_rate))
    score = roc_auc_score(y_test, model.predict(X_text))
    return score

def record_model(model_name, model, score):
    model_inst = {}
    model_inst['model'] = model
    model_inst['model_name'] = model_name
    model_inst['score'] = score
    print('{}: {}'.format(model_name, score))
    models.append(model_inst)

def run(df, train=True):
    print('1')
    df = handle_categoricals(df)
    print('2')
    df = drop_nulls(df)
    if train:
        print('3')
        X_train, X_test, y_train, y_test = gen_train_test_split(df)
        model = model_rf(X_train, y_train)
        score = score_model(model, X_test, y_test)
        record_model('random_forest',model, score)
    else:
        print('4')
        score = 0
        selected_model = None
        for model_inst in models:
            if model_inst['score'] > score:
                score = model_inst['score']
                selected_model = model_inst['model']
        prediction = selected_model.predict(df)
        print(prediction)

run(df_application, True)
run(df_application_test, False)