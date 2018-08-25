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
from multiprocessing import Process
import multiprocessing as mp
from collections import Counter

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

            if str(df[col].dtype) in ['object', 'category']:
                # If 2 or fewer unique categories
                if len(list(df[col].unique())) <= 2:
                    #print('le :' + col)
                    # Train on the training data
                    le.fit(all_df[col].astype(str))
                    # Transform both training and testing data
                    df['le_'+str(col)] = le.transform(df[col].astype(str))
                    # df_application[col] = le.transform(df_application[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1
                else:
                    oh_count +=1
                    #print('ce :' + col)
                    oh_df = pd.concat([oh_df,pd.get_dummies(df[col].astype(str),prefix=col+'_')],axis=1)
                df.drop([col], axis=1, inplace=True)

        print('%d columns were label encoded.' % le_count)
        print('%d columns were one hot encoded.' % oh_count)

        df= pd.concat([df,oh_df],axis=1)
        return df

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

    def compute_score(model, X_test, y_test):
        #predictions = rf_model.predict(test_features)
        #accuracy =  compute_score(rf_model,test_features,test_labels)
        #print(accuracy)
        #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        #print(auc(false_positive_rate, true_positive_rate))
        score = roc_auc_score(y_test, model.predict(X_test))
        return score

    def record_model(model_name, model, score):
        model_inst = {}
        model_inst['model'] = model
        model_inst['model_name'] = model_name
        model_inst['score'] = score
        print('Recording Model and Score ===> {}: {}'.format(model_name, score))
        models.append(model_inst)


class Models:
    def model_rf(train_features, test_features, train_labels, test_labels):
        print('running Random Forest')
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
        print('running gbm')
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
        print('running xgboost')
        import xgboost as xgb
        xgb_model_def = xgb.XGBClassifier(max_depth=20,
                                          n_estimators=100,
                                          learning_rate=0.01,
                                          random_state=42,
                                          max_features='sqrt',
                                          nthread=-1)
        xgb_model = xgb_model_def.fit(train_features,train_labels)
        accuracy = util.compute_score(xgb_model, test_features, test_labels)
        util.record_model("xgb_1", xgb_model, accuracy)

    def model_lgbm(X,X_test, y, y_test):
        print('running light gbm')
        # print(X.shape)
        # print(X_test.shape)
        # X.fillna(0, inplace=True)
        # X_test  .fillna(0, inplace=True)
        # y.fillna(0, inplace=True)
        # y_test.fillna(0,inplace = True)
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X = sc.fit_transform(X)
        # X_test = sc.transform(X_test)
        params = {}
        categorical_features=[]
        #categorical_features = [c for c, col in enumerate(X.columns) if 'cat' in col]
        train_data = lgbm.Dataset(X, label=y, categorical_feature=categorical_features)
        test_data = lgbm.Dataset(X_test, label=y_test)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'metric_freq': 1,
            'is_training_metric': True,
            'max_bin': 255,
            'learning_rate': 0.1,
            'num_leaves': 63,
            'tree_learner': 'serial',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'min_sum_hessian_in_leaf': 5,
            'is_enable_sparse': True,
            'use_two_round_loading': False,
            'is_save_binary_file': False,
            'output_model': 'LightGBM_model.txt',
            'num_machines': 1,
            'local_listen_port': 12400,
            'machine_list_file': 'mlist.txt',
            'verbose': 0,
            'subsample_for_bin': 200000,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0
        }
        # Create parameters to search
        gridParams = {
            'learning_rate': [0.1],
            'num_leaves': [63],
            'boosting_type': ['gbdt'],
            'objective': ['binary']
        }
        #
        model = lgbm.LGBMClassifier(
            task = params['task'],
            metric = params['metric'],
            metric_freq = params['metric_freq'],
            is_training_metric = params['is_training_metric'],
            max_bin = params['max_bin'],
            tree_learner = params['tree_learner'],
            feature_fraction = params['feature_fraction'],
            bagging_fraction = params['bagging_fraction'],
            bagging_freq = params['bagging_freq'],
            min_data_in_leaf = params['min_data_in_leaf'],
            min_sum_hessian_in_leaf = params['min_sum_hessian_in_leaf'],
            is_enable_sparse = params['is_enable_sparse'],
            use_two_round_loading = params['use_two_round_loading'],
            is_save_binary_file = params['is_save_binary_file'],
            n_jobs = -1)

        #
        scoring = {'AUC': 'roc_auc'}

        # Create the grid

        # Run the grid
        grid.fit(X, y)
        params = grid.best_params_


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
        # Feature Scaling

        # params['learning_rate'] = 0.003
        # params['boosting_type'] = 'gbdt'
        # params['objective'] = 'binary'
        # params['metric'] = 'binary_logloss'
        # params['sub_feature'] = 0.5
        # params['num_leaves'] = 10
        # params['min_data'] = 50
        # params['max_depth'] = 10

        model = lgbm.train(params,
                           train_data,
                           valid_sets=[test_data],
                           verbose_eval=-1)
        util.record_model("lgbm_1", model, 10)









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

binary_binnable_cols =[
'OBS_30_CNT_SOCIAL_CIRCLE',
'DEF_30_CNT_SOCIAL_CIRCLE',
'OBS_60_CNT_SOCIAL_CIRCLE',
'DEF_60_CNT_SOCIAL_CIRCLE',
'AMT_REQ_CREDIT_BUREAU_HOUR',
'AMT_REQ_CREDIT_BUREAU_DAY',
'AMT_REQ_CREDIT_BUREAU_WEEK',
'AMT_REQ_CREDIT_BUREAU_MON',
'AMT_REQ_CREDIT_BUREAU_QRT',
'AMT_REQ_CREDIT_BUREAU_YEAR']

def clean_up(df):
    df = util.handle_nulls(df)
    return df

def fe(df, df_2):
    df['HAS_CHILDREN']= binn_col(df,'CNT_CHILDREN',[-1,1,50], labels=[0,1])
    df['binned_' +'CNT_CHILDREN']= binn_col(df,'CNT_CHILDREN',[-1,1,3,50], labels=[1,2,3])
    for col in array_binnable_cols:
        #print(col)
        df['binned_' +col] = bin_col_q_bins(df, col, 10)
        #df['binned_' +col] = bin_col_q_bins(df, col, 10)

    binary_binns = [-1,1,10000]
    binary_bin_labels = [0,1]
    for col in binary_binnable_cols:
        # print(col)
        df['binary_binned_' + col] = binn_col(df, col, binary_binns, binary_bin_labels)

    df['NAME_FAMILY_STATUS_IS_MARRIED'] = df['NAME_FAMILY_STATUS'].map({'Civilmarriage':1, 'Married':1, 'Separated':0, 'Single/notmarried':0,'Unknown':0, 'Widow':0})
    df['NAME_INCOME_TYPE_IS_WORKING'] = df['NAME_INCOME_TYPE'].map({'Businessman':0,'Commercialassociate':0,'Maternityleave':0, 'Pensioner':0,'Stateservant':1, 'Student':0,'Unemployed':0, 'Working':1})
    df['NAME_EDUCATION_TYPE_IS_WELL_EDUCATED'] = df['NAME_EDUCATION_TYPE'].map({'Academicdegree':1,'Highereducation':1,'Incompletehigher':1,'Lowersecondary':0,'Secondary/secondaryspecial':0})



    # feature interactions
    df['annuity_income_percentage'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['income_per_child'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['phone_to_birth_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['phone_to_employ_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']

    # External sources# Exter
    # df['external_sources_weighted'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
    # for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
    #     df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
    #         df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    #
    # AGGREGATION_RECIPIES = [
    #     (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
    #                                               ('AMT_CREDIT', 'max'),
    #                                               ('EXT_SOURCE_1', 'mean'),
    #                                               ('EXT_SOURCE_2', 'mean'),
    #                                               ('OWN_CAR_AGE', 'max'),
    #                                               ('OWN_CAR_AGE', 'sum')]),
    #     (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
    #                                             ('AMT_INCOME_TOTAL', 'mean'),
    #                                             ('DAYS_REGISTRATION', 'mean'),
    #                                             ('EXT_SOURCE_1', 'mean')]),
    #     (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
    #                                                  ('CNT_CHILDREN', 'mean'),
    #                                                  ('DAYS_ID_PUBLISH', 'mean')]),
    #     (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
    #                                                                                            ('EXT_SOURCE_2',
    #                                                                                             'mean')]),
    #     (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
    #                                                   ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
    #                                                   ('APARTMENTS_AVG', 'mean'),
    #                                                   ('BASEMENTAREA_AVG', 'mean'),
    #                                                   ('EXT_SOURCE_1', 'mean'),
    #                                                   ('EXT_SOURCE_2', 'mean'),
    #                                                   ('EXT_SOURCE_3', 'mean'),
    #                                                   ('NONLIVINGAREA_AVG', 'mean'),
    #                                                   ('OWN_CAR_AGE', 'mean'),
    #                                                   ('YEARS_BUILD_AVG', 'mean')]),
    #     (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
    #                                                                             ('EXT_SOURCE_1', 'mean')]),
    #     (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
    #                            ('CNT_CHILDREN', 'mean'),
    #                            ('CNT_FAM_MEMBERS', 'mean'),
    #                            ('DAYS_BIRTH', 'mean'),
    #                            ('DAYS_EMPLOYED', 'mean'),
    #                            ('DAYS_ID_PUBLISH', 'mean'),
    #                            ('DAYS_REGISTRATION', 'mean'),
    #                            ('EXT_SOURCE_1', 'mean'),
    #                            ('EXT_SOURCE_2', 'mean'),
    #                            ('EXT_SOURCE_3', 'mean')]),
    # ]

    # groupby_aggregate_names = []
    # for groupby_cols, specs in AGGREGATION_RECIPIES:
    #     group_object = df.groupby(groupby_cols)
    #     for select, agg in specs:
    #         groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
    #         df = df.merge(group_object[select]
    #                     .agg(agg)
    #                     .reset_index()
    #                     .rename(index=str,
    #                             columns={select: groupby_aggregate_name})
    #                     [groupby_cols + [groupby_aggregate_name]],
    #                     on=groupby_cols,
    #                     how='left')
    #         groupby_aggregate_names.append(groupby_aggregate_name)
    #
    # diff_feature_names = []
    # for groupby_cols, specs in AGGREGATION_RECIPIES:
    #     for select, agg in specs:
    #         if agg in ['mean', 'median', 'max', 'min']:
    #             groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
    #             diff_name = '{}_diff'.format(groupby_aggregate_name)
    #             abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)
    #
    #             df[diff_name] = df[select] - df[groupby_aggregate_name]
    #             df[abs_diff_name] = np.abs(df[select] - df[groupby_aggregate_name])
    #
    #             diff_feature_names.append(diff_name)
    #             diff_feature_names.append(abs_diff_name)




    df = util.handle_categoricals(df, pd.concat([df, df_2],axis=0))

    return df

def align_datasets(df, df2):
    df = util.add_missing_dummy_columns(df, df2.columns)
    df2 = util.add_missing_dummy_columns(df2, df.columns)

    df.columns = df.columns.str.replace(' ', '')
    df2.columns =  df2.columns.str.replace(' ','')
    df = util.handle_nulls(df)
    df2 = util.handle_nulls(df2)
    return df, df2


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

def run(df, df_test, train=True):

    print('splitting data')
    X_train, X_test, y_train, y_test = util.gen_train_test_split(df)

    Models.model_rf(X_train, X_test,y_train, y_test)
    #Models.model_gbm(X_train, X_test, y_train, y_test)
    #Models.model_xgb_2(X_train, X_test, y_train, y_test)
    Models.model_lgbm(X_train, X_test, y_train, y_test)
    score = 0
    selected_model = None
    print(models)
    for model_inst in models:
        if model_inst['score'] > score:
            name = model_inst['model_name']
            score = model_inst['score']
            selected_model = model_inst['model']
        print(name + " :  " + str(score))

    #prediction = selected_model.predict(df_test_raw.drop(['TARGET'],axis=1))
    df_test=df_test[X_train.columns]
    prediction = selected_model.predict(df_test)
    submission_df = pd.DataFrame()
    submission_df['SK_ID_CURR'] = df_test['SK_ID_CURR']
    submission_df['TARGET'] = prediction
    submission_df.to_csv('home_credit_submission.csv',',', index=False)

def make_subset(df, subset=0):
    if subset >0:
        df = df[:subset]
    return df

df_application = None
df_application_test = None
arr_in_process_cols=[]
arr_in_process_cols_test=[]

output = mp.Queue()
def prep():
    # load data
    df_application, df_application_test = load_data()


    # Make Subset :
    subset = 0
    df_application = make_subset(df_application, subset)
    df_application_test = make_subset(df_application_test, subset)
    print("{},{}".format(df_application.shape, df_application_test.shape))

    # Clean up data
    df_application = clean_up(df_application)
    df_application_test =  clean_up(df_application_test)
    print("{},{}".format(df_application.shape, df_application_test.shape))
    # # Feature Engineer
    #df_application = fe(df_application, df_application_test)
    #df_application_test = fe(df_application_test, df_application)
    # print("{},{}".format(df_application.shape, df_application_test.shape))
    # print(df_application['TARGET'].unique())

    parallel_funcs =[]
    # Define an output queue


    df_installments_payments = pd.read_csv('installments_payments.csv')
    df_bureau = pd.read_csv('bureau.csv')
    df_pos_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    df_prev_application = pd.read_csv('previous_application.csv')
    df_credit_card_balance = pd.read_csv('credit_card_balance.csv')

    #df_application['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    df_bureau['DAYS_CREDIT_ENDDATE'][df_bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    df_bureau['DAYS_CREDIT_UPDATE'][df_bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    df_bureau['DAYS_ENDDATE_FACT'][df_bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

    df_credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'][df_credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    df_credit_card_balance['AMT_DRAWINGS_CURRENT'][df_credit_card_balance['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

    parallel_funcs.append({'id':1, 'function1': get_agg_numerics_data, 'args1':[df_application, df_application,['SK_ID_CURR'], output,True]})
    parallel_funcs.append({'id':2, 'function1': get_agg_numerics_data, 'args1':[df_application_test, df_application_test, ['SK_ID_CURR'], output,False]})
    parallel_funcs.append({'id':3, 'function1': get_agg_numerics_data, 'args1':[df_application,'installments_payments', pd.read_csv('installments_payments.csv'),['SK_ID_CURR'], output,True]})
    parallel_funcs.append({'id':4, 'function1': get_agg_numerics_data, 'args1': [df_application_test,'installments_payments', pd.read_csv('installments_payments.csv'), ['SK_ID_CURR'], output,False]})
    parallel_funcs.append({'id':5, 'function1': get_agg_numerics_data, 'args1':[df_application,'bureau', pd.read_csv('bureau.csv'),['SK_ID_CURR'], output,True]})
    parallel_funcs.append({'id':6, 'function1': get_agg_numerics_data, 'args1':[df_application_test, 'bureau', pd.read_csv('bureau.csv'), ['SK_ID_CURR'], output,False]})
    parallel_funcs.append({'id':7, 'function1': get_agg_numerics_data, 'args1':[df_application,'POS_CASH_balance', pd.read_csv('POS_CASH_balance.csv'), ['SK_ID_CURR'], output,True]})
    parallel_funcs.append({'id':8, 'function1': get_agg_numerics_data, 'args1':[df_application_test,'POS_CASH_balance', pd.read_csv('POS_CASH_balance.csv'), ['SK_ID_CURR'], output,False]})
    parallel_funcs.append({'id':9, 'function1': get_agg_numerics_data, 'args1':[df_application,'previous_application', pd.read_csv('previous_application.csv'), ['SK_ID_CURR'], output,True]})
    parallel_funcs.append({'id':10, 'function1': get_agg_numerics_data, 'args1':[df_application_test,'previous_application', pd.read_csv('previous_application.csv'), ['SK_ID_CURR'],output,False]})
    parallel_funcs.append({'id':11, 'function1': get_agg_numerics_data, 'args1':[df_application,'POS_CASH_balance', pd.read_csv('POS_CASH_balance.csv'), ['SK_ID_CURR'],output,True]})
    parallel_funcs.append({'id':12, 'function1': get_agg_numerics_data, 'args1':[df_application_test,'POS_CASH_balance', pd.read_csv('POS_CASH_balance.csv'), ['SK_ID_CURR'],output,False]})

    arr_in_process_cols = []
    arr_in_process_cols_test = []
    arr_in_process_cols.append(df_application)
    arr_in_process_cols_test.append(df_application_test)

    arr_in_process_cols_tmp, arr_in_process_cols_test_tmp = runInParallel(parallel_funcs)
    print(3)

    print(len(arr_in_process_cols_tmp))
    print(len(arr_in_process_cols_test_tmp))

    for df_tmp in arr_in_process_cols_tmp:
        df_application = pd.concat([df_application, df_tmp],axis=1)
    df_tmp = None
    for df_tmp in arr_in_process_cols_test_tmp:
        df_application_test = pd.concat([df_application_test, df_tmp],axis=1)

    print("{},{}".format(df_application.shape, df_application_test.shape))



    df_application= duplicate_columns(df_application)
    df_application_test= duplicate_columns(df_application_test)

    df_application = df_application.reset_index()
    df_application_test = df_application_test.reset_index()
    df_application = df_application.replace([np.inf, -np.inf], 0)
    df_application_test = df_application_test.replace([np.inf, -np.inf], 0)

    # Align Datasets
    df_application, df_application_test = align_datasets(df_application,df_application_test)

    print("{},{}".format(df_application.shape, df_application_test.shape))

    print(Counter(df_application))
    # Checkpoint Save
    df_application.to_pickle('application_processed.pickle')
    df_application_test.to_pickle('application_test_processed.pickle')
    print("{},{}".format(df_application.shape, df_application_test.shape))


def duplicate_columns(frame):
    arr_cols = []
    arr_col_ind =[]
    df= pd.DataFrame()
    for id in range(len(frame.columns)-1):
        print(id)
        if frame.iloc[:, id].name not in arr_cols:
            #df= pd.concat([df, frame.iloc[:,id]], axis=1)
            arr_col_ind.append(id)
            arr_cols.append(frame.iloc[:, id].name)
    print(arr_col_ind)
    df = pd.concat([df, frame.iloc[:, arr_col_ind]], axis=1)
    return df

def runInParallel(fns):
  proc = []
  for fn in fns:
    p = Process(target=fn['function1'],args=fn['args1'])
    p.start()
    proc.append(p)
  b=[]
  b_test=[]
  for p in proc:
    results = output.get()
    #print(results)

    a = results[0]
    a_test =results[1]

    b.append(a)
    b_test.append(a_test)
  for p in proc:
    p.join()
  return b, b_test

def get_agg_numerics_data(df, file_name,  additional_data, exclude_cols, output, train= True):
    arr_in_process_cols=None
    arr_in_process_cols_test=None
    numeric_columns = []
    for col in additional_data.columns:
        if str(additional_data[col].dtype) not in ['object','category'] and col not in exclude_cols:
            numeric_columns.append(col)
    AGGREGATION_RECIPIES = []
    for agg in ['mean', 'min', 'max', 'sum', 'var']:
        for select in numeric_columns:
            AGGREGATION_RECIPIES.append((select, agg))
    AGGREGATION_RECIPIES = [(['SK_ID_CURR'], AGGREGATION_RECIPIES)]

    groupby_aggregate_names = []
    counter =0
    for groupby_cols, specs in AGGREGATION_RECIPIES:
        group_object = additional_data.groupby(groupby_cols)
        for select, agg in specs:
            counter +=1
            groupby_aggregate_name = '{}_{}_{}_{}_{}'.format(str(counter),file_name, '_'.join(groupby_cols), agg, select)
            df = df.merge(group_object[select]
                                  .agg(agg)
                                  .reset_index()
                                  .rename(index=str,
                                          columns={select: groupby_aggregate_name})
                                  [groupby_cols + [groupby_aggregate_name]],
                                  on=groupby_cols,
                                  how='inner')
            groupby_aggregate_names.append(groupby_aggregate_name)
    if train:
        arr_in_process_cols= df[list(set(groupby_aggregate_names))]
    else:
        arr_in_process_cols_test = df[list(set(groupby_aggregate_names))]

    output.put((arr_in_process_cols,arr_in_process_cols_test))
        
    print("{}".format(df.shape))



def load_data():
    df_application = pd.read_csv('application_train.csv')
    df_application_test = pd.read_csv('application_test.csv')
    # df_bureau_balance = pd.read_csv('bureau_balance.csv')
    # df_installment_payments = pd.read_csv('installments_payments.csv')
    # df_bureau = pd.read_csv('bureau.csv')
    # df_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    # df_prev_application = pd.read_csv('previous_application.csv')
    # return df_application,df_application_test, df_bureau, df_bureau_balance, df_cash_balance, df_installment_payments, df_prev_application
    return df_application, df_application_test
    # Feature Selection
    #df_application, df_application_test = featureSelection(df_application, df_application_test)


prep()
exit()





df = pd.read_pickle('application_processed.pickle')
df_test = pd.read_pickle('application_test_processed.pickle')

print(Counter(df))
print(df.shape)
print(df_test.shape)
print(set(df.columns)-set(df_test.columns))
#df.drop(['CODE_GENDER__XNA'],axis=1, inplace=True)


run(df,df_test,True)
