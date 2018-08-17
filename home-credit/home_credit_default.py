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
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings("ignore")

models = []


class util:

    def handle_categoricals(df, all_data):
        # Create a label encoder object


        le_count = 0
        oh_count = 0
        oh_df = pd.DataFrame()
        # Iterate through the columns
        for col in df:
            le = LabelEncoder()
            if col == 'TARGET':
                continue
            all_df = pd.DataFrame()
            all_df[col] = all_data[col]

            if df[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(df[col].unique())) <= 2:
                    print('le :' + col)
                    # Train on the training data
                    le.fit(all_df[col].astype(str))
                    # Transform both training and testing data
                    df[col] = le.transform(df[col])
                    # df_application[col] = le.transform(df_application[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1
                else:
                    oh_count +=1
                    print('ce :' + col)
                    oh_df = pd.concat([oh_df,pd.get_dummies(df[col].astype(str),prefix=col+'_')],axis=1)
                    df.drop([col], axis=1, inplace=True)
                    print(oh_df.columns.values)

        print('%d columns were label encoded.' % le_count)
        print('%d columns were one hot encoded.' % oh_count)

        df= pd.concat([df,oh_df],axis=1)
        return df

    def add_missing_dummy_columns( d, columns ):
        missing_cols = set( columns ) - set( d.columns )
        for c in missing_cols:
            d[c] = 0
        return d

    def handle_nulls(df):
        for col in df:
            if col == 'TARGET':
                continue
            #print(df[col].dtype)
            if df[col].dtype == 'object':
                #df[col].add_categories(['un_specified'])
                df[col] = df[col].fillna('un_specified')
            elif str(df[col].dtype) == 'category':
                print(str(df[col].dtype))
                df[col] = df[col].cat.add_categories([999]).fillna(999)
            else:
                df[col] = df[col].fillna(999)
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
        train_test_split(X, y, test_size=0.2, random_state=42)
        return train_features, test_features, train_labels, test_labels

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


class Models:
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
        score = util.compute_score(model, test_features, test_labels)
        util.record_model('random_forest',model, score)
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
        accuracy = util.compute_score(gb, test_features, test_labels)
        util.record_model("gbm_1", gb, accuracy)

    def model_xgb_2(train_features, test_features, train_labels, test_labels):
        import xgboost as xgb
        xgb_model_def = xgb.XGBClassifier(max_depth=20,
                                          n_estimators=100,
                                          learning_rate=0.01,
                                          random_state=42,
                                          max_features='sqrt')
        xgb_model = xgb_model_def.fit(train_features,train_labels)
        accuracy = util.compute_score(xgb_model, test_features, test_labels)
        util.record_model("xgb_1", xgb_model, accuracy)

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
            'learning_rate': [0.001],
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
        util.record_model("lgbm_1", model, 10)







df_application = pd.read_csv('application_train.csv')
df_application_test = pd.read_csv('application_test.csv')
df_application = df_application
df_application_test = df_application_test

# CNT_CHILDREN
def binn_col(df,col,buckets,labels):
    return pd.cut(df[col],buckets, labels=labels)


def bin_col_q_bins(df,col, bins=10):
    return pd.qcut(df[col],bins, labels=False,duplicates='drop')



array_binnable_cols = ['AMT_CREDIT',
'AMT_ANNUITY',
'AMT_GOODS_PRICE',
'REGION_POPULATION_RELATIVE',
'DAYS_BIRTH',
'DAYS_EMPLOYED',
'DAYS_REGISTRATION',
'DAYS_ID_PUBLISH',
'OWN_CAR_AGE',
'APARTMENTS_AVG',
'BASEMENTAREA_AVG',
'YEARS_BEGINEXPLUATATION_AVG',
'YEARS_BUILD_AVG',
'COMMONAREA_AVG',
'ELEVATORS_AVG',
'ENTRANCES_AVG',
'FLOORSMAX_AVG',
'FLOORSMIN_AVG',
'LANDAREA_AVG',
'LIVINGAPARTMENTS_AVG',
'LIVINGAREA_AVG',
'NONLIVINGAPARTMENTS_AVG',
'NONLIVINGAREA_AVG',
'APARTMENTS_MODE',
'BASEMENTAREA_MODE',
'YEARS_BEGINEXPLUATATION_MODE',
'YEARS_BUILD_MODE',
'COMMONAREA_MODE',
'ELEVATORS_MODE',
'ENTRANCES_MODE',
'FLOORSMAX_MODE',
'FLOORSMIN_MODE',
'LANDAREA_MODE',
'LIVINGAPARTMENTS_MODE',
'LIVINGAREA_MODE',
'NONLIVINGAPARTMENTS_MODE',
'NONLIVINGAREA_MODE',
'APARTMENTS_MEDI',
'BASEMENTAREA_MEDI',
'YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BUILD_MEDI',
'COMMONAREA_MEDI',
'ELEVATORS_MEDI',
'ENTRANCES_MEDI',
'FLOORSMAX_MEDI',
'FLOORSMIN_MEDI',
'LANDAREA_MEDI',
'LIVINGAPARTMENTS_MEDI',
'LIVINGAREA_MEDI',
'NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAREA_MEDI']



def clean_up(df):
    df = util.handle_nulls(df)
    return df

def fe(df, df_2):
    df['HAS_CHILDREN']= binn_col(df,'CNT_CHILDREN',[-1,1,50], labels=[0,1])
    df['binned_' +'CNT_CHILDREN']= binn_col(df,'CNT_CHILDREN',[-1,1,3,50], labels=[1,2,3])
    for col in array_binnable_cols:
        print(col)
        df['binned_' +col] = bin_col_q_bins(df, col, 10)
        df['binned_' +col] = bin_col_q_bins(df, col, 10)


    df['NAME_FAMILY_STATUS_IS_MARRIED'] = df['NAME_FAMILY_STATUS'].map({'Civilmarriage':1, 'Married':1, 'Separated':0, 'Single/notmarried':0,'Unknown':0, 'Widow':0})
    df['NAME_INCOME_TYPE_IS_WORKING'] = df['NAME_INCOME_TYPE'].map({'Businessman':0,'Commercialassociate':0,'Maternityleave':0, 'Pensioner':0,'Stateservant':1, 'Student':0,'Unemployed':0, 'Working':1})
    df['NAME_EDUCATION_TYPE_IS_WELL_EDUCATED'] = df['NAME_EDUCATION_TYPE'].map({'Academicdegree':1,'Highereducation':1,'Incompletehigher':1,'Lowersecondary':0,'Secondary/secondaryspecial':0})

    df = util.handle_categoricals(df, pd.concat([df, df_2],axis=0))

    return df

def align_datasets(df, df2):
    df = util.add_missing_dummy_columns(df, df2.columns)
    df.columns = df.columns.str.replace(' ', '')
    return df


def featureSelection(df, df_test):
    # Feature Selection
    model = ExtraTreesClassifier()
    model.fit(df.drop(['TARGET'], axis=1), df['TARGET'])
    fi =model.feature_importances_
    features_with_imp = dict(zip(df.columns.values, fi))
    fi.sort()
    top_feature_importance = fi[int(len(fi)*.2):len(fi)-1]
    min_fi = min(top_feature_importance)
    selected_features = list((k for (k, v) in features_with_imp.items() if (v > min_fi) or k=='TARGET'))
    print(selected_features)
    df = df[selected_features]
    df_test = df_test[selected_features]
    return df, df_test

def run(df, df_test, df_all, train=True):
    df_raw= df
    df_test_raw= df_test
    X_train, X_test, y_train, y_test = util.gen_train_test_split(df)

    #model_rf(X_train, X_test,y_train, y_test)
    #model_gbm(X_train, X_test, y_train, y_test)
    #model_xgb_2(X_train, X_test, y_train, y_test)
    #Models.model_lgbm(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    Models.model_lgbm(X_train, X_test, y_train, y_test)
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
    prediction = selected_model.predict(df_test)
    submission_df = pd.DataFrame()
    submission_df['SK_ID_CURR'] = df_test['SK_ID_CURR']
    submission_df['TARGET'] = prediction
    submission_df.to_csv('home_credit_submission.csv',',', index=False)

def make_subset(df, subset=0):
    if subset >0:
        df = df[:subset]
    return df





# Make Subset
import  sys
if len(sys.argv) >2:
    subset = int(sys.argv[1])
    df_application = make_subset(df_application, subset)
    df_application_test = make_subset(df_application_test, subset)

# Clean up data
df_application = clean_up(df_application)
df_application_test =  clean_up(df_application_test)
# Feature Engineer
df_application = fe(df_application, df_application_test)
df_application_test = fe(df_application_test, df_application)
# Align Datasets
df_application = align_datasets(df_application,df_application_test)
df_application = align_datasets(df_application_test,df_application)
# Checkpoint Save
df_application.to_csv('application_processed.csv')
df_application_test.to_csv('application_processed.csv')
# Feature Selection
#df_application, df_application_test = featureSelection(df_application, df_application_test)

# Run Models
all_data = pd.concat([df_application, df_application_test],axis=0)
run(df_application,df_application_test,all_data, True)
