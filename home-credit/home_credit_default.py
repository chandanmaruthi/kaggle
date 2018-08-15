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
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV


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
                df.drop([col], axis=1, inplace=True)

    print('%d columns were label encoded.' % le_count)

    df= pd.concat([df,oh_df],axis=1)
    #print(df.columns)
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

def model_rf(train_features, test_features, train_labels, test_labels):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 100,
                                      random_state = 42,
                                      n_jobs=-1,
                                      oob_score=True,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features='auto',
                                      bootstrap=True)
    # Train the model on training data
    model.fit(train_features, train_labels)
    score = compute_score(model, test_features, test_labels)
    record_model('random_forest',model, score)
    return

def model_gbm(train_features, test_features, train_labels, test_labels):
    from sklearn.ensemble import GradientBoostingClassifier
    param_test1 = {'n_estimators':range(20,81,10)}

    gb = GradientBoostingClassifier(criterion='mse',
                  learning_rate=0.01,
                  max_depth=10,
                  n_estimators=1000,
                  random_state=42,
                  max_features='sqrt')
    gb.fit(train_features,train_labels)
    accuracy = compute_score(gb, test_features, test_labels)
    record_model("gbm_1", gb, accuracy)

def model_xgb_2(train_features, test_features, train_labels, test_labels):
    import xgboost as xgb
    xgb_model_def = xgb.XGBClassifier(max_depth=20,
                                      n_estimators=100,
                                      learning_rate=0.01,
                                      random_state=42,
                                      max_features='sqrt')
    xgb_model = xgb_model_def.fit(train_features,train_labels)
    accuracy = compute_score(xgb_model, test_features, test_labels)
    record_model("xgb_1", xgb_model, accuracy)

def model_lgbm(X,X_test, y, y_test):
    categorical_features = [c for c, col in enumerate(X.columns) if 'cat' in col]
    train_data = lgbm.Dataset(X, label=y, categorical_feature=categorical_features)
    test_data = lgbm.Dataset(X_test, label=y_test)

    params = {'boosting_type': 'gbdt',
              'max_depth': -1,
              'objective': 'binary',
              'nthread': 3,  # Updated from nthread
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class': 1,
              'metric': 'auc'}
    # Create parameters to search
    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [40],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [501],  # Updated from 'seed'
        'colsample_bytree': [0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
    }

    model = lgbm.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

    grid = GridSearchCV(model, gridParams,
                        verbose=1,
                        cv=4,
                        n_jobs=2)


    # Run the grid
    grid.fit(X, y)
    params = grid.best_params_
    print(grid.best_params_)
    print(grid.best_score_)
    # exit()
    # def prepLGB(data,
    #             classCol='',
    #             IDCol='',
    #             fDrop=[]):
    #
    #     # Drop class column
    #     if classCol != '':
    #         labels = data[classCol]
    #         fDrop = fDrop + [classCol]
    #     else:
    #         labels = []
    #
    #     if IDCol != '':
    #         IDs = data[IDCol]
    #     else:
    #         IDs = []
    #
    #     if fDrop != []:
    #         data = data.drop(fDrop,
    #                          axis=1)
    #
    #     # Create LGB mats
    #     lData = lgbm.Dataset(data, label=labels,
    #                         free_raw_data=False,
    #                         feature_name=list(data.columns),
    #                         categorical_feature='auto')
    #
    #     return lData, labels, IDs, data
    #
    # params = {'boosting_type': 'gbdt', 'colsample_bytree': 0.65, 'learning_rate': 0.005, 'n_estimators': 40, 'num_leaves': 6,
    #  'objective': 'binary', 'random_state': 501, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.7}
    model = lgbm.train(params,
                    train_data,
                    num_boost_round=10000,
                    valid_sets=[train_data, test_data],
                    early_stopping_rounds=500,
                    verbose_eval=4)
    #
    #accuracy = compute_score(model, X_test, y_test)
    record_model("lgbm_1", model, 10)


def compute_score(model, X_test, y_test):
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
    df_raw= df
    df_test_raw= df_test
    print('1')
    df = handle_categoricals(df, df_all)
    print('2')
    #df = handle_nulls(df)
    df.fillna(0, inplace=True)
    df_test = handle_categoricals(df_test,df_all)

    print(df.shape)
    print(df_test.shape)
    df = add_missing_dummy_columns(df, df_test.columns)
    df_test = add_missing_dummy_columns(df_test, df.columns)
    print(df.shape)
    print(df.shape)
    df_test.fillna(0, inplace=True)
    #exit()
    X_train, X_test, y_train, y_test = gen_train_test_split(df)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = gen_train_test_split(df_raw)
    #model_rf(X_train, X_test,y_train, y_test)
    #model_gbm(X_train, X_test, y_train, y_test)
    #model_xgb_2(X_train, X_test, y_train, y_test)
    model_lgbm(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    #exit()
    #score = score_model(model, X_test, y_test)
    #record_model('random_forest',model, score)
    #model = model_lgbm(X_train, y_train)
    #y_pred = model.predict(X_test)
    #score = roc_auc_score(y_test, model.predict(X_test))

    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict(X_test))
    #score = score_model(model, X_test, y_test)
    #record_model('lgbm',model, score)
    print('4')
    score = 0

    selected_model = None

    print(models)
    for model_inst in models:
        if model_inst['score'] > score:
            name = model_inst['model_name']
            score = model_inst['score']
            selected_model = model_inst['model']
        print(str(score) + '_' + name)

    #prediction = selected_model.predict(df_test_raw.drop(['TARGET'],axis=1))
    prediction = selected_model.predict(df_test_raw)
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
df_application = df_application
df_application_test = df_application_test


df_application = df_application
df_application_test = df_application_test

# le = LabelEncoder()
# le.fit(df_application['NAME_TYPE_SUITE'].astype(str))
# df_tmp = le.transform(df_application['NAME_TYPE_SUITE'].astype(str))
# ohe=OneHotEncoder()
# ohe.fit(df_tmp)

all_data = pd.concat([df_application, df_application_test],axis=0)
run(df_application,df_application_test,all_data, True)
#run(df_application, df_application_test, all_data, False)