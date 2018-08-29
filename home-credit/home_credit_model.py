import matplotlib
import numpy as np
# 1import matplotlib.pyplot as plt
# %matplotlib inline
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

    def compute_score(clf, X, y, scoring='accuracy'):
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        xval = cross_val_score(clf, X, y, cv=10, scoring=scoring)
        return np.mean(xval)

    def compute_score(model, X_test, y_test):
        # predictions = rf_model.predict(test_features)
        # accuracy =  compute_score(rf_model,test_features,test_labels)
        # print(accuracy)
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        # print(auc(false_positive_rate, true_positive_rate))
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
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=42,
                                       n_jobs=-1,
                                       oob_score=True,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       max_features='auto',
                                       bootstrap=True)

        # Train the model on training data
        model.fit(train_features, train_labels)

        score = util.compute_score(model, test_features, test_labels)
        util.record_model('random_forest', model, score)
        return

    def model_gbm(train_features, test_features, train_labels, test_labels):
        print('running gbm')
        from sklearn.ensemble import GradientBoostingClassifier
        param_test1 = {'n_estimators': range(20, 81, 10)}

        gb = GradientBoostingClassifier(criterion='mse',
                                        learning_rate=0.01,
                                        max_depth=10,
                                        n_estimators=1000,
                                        random_state=42,
                                        max_features='sqrt')
        gb.fit(train_features, train_labels)
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
        xgb_model = xgb_model_def.fit(train_features, train_labels)
        accuracy = util.compute_score(xgb_model, test_features, test_labels)
        util.record_model("xgb_1", xgb_model, accuracy)

    def model_lgbm(X, X_test, y, y_test):
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
        categorical_features = []
        # categorical_features = [c for c, col in enumerate(X.columns) if 'cat' in col]
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
            task=params['task'],
            metric=params['metric'],
            metric_freq=params['metric_freq'],
            is_training_metric=params['is_training_metric'],
            max_bin=params['max_bin'],
            tree_learner=params['tree_learner'],
            feature_fraction=params['feature_fraction'],
            bagging_fraction=params['bagging_fraction'],
            bagging_freq=params['bagging_freq'],
            min_data_in_leaf=params['min_data_in_leaf'],
            min_sum_hessian_in_leaf=params['min_sum_hessian_in_leaf'],
            is_enable_sparse=params['is_enable_sparse'],
            use_two_round_loading=params['use_two_round_loading'],
            is_save_binary_file=params['is_save_binary_file'],
            n_jobs=-1)

        #
        scoring = {'AUC': 'roc_auc'}

        # Create the grid

        # Run the grid
        # grid.fit(X, y)
        # params = grid.best_params_

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


def run(df, df_test, train=True):
    print('splitting data')
    X_train, X_test, y_train, y_test = util.gen_train_test_split(df)

    Models.model_rf(X_train, X_test, y_train, y_test)
    # Models.model_gbm(X_train, X_test, y_train, y_test)
    # Models.model_xgb_2(X_train, X_test, y_train, y_test)
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

    # prediction = selected_model.predict(df_test_raw.drop(['TARGET'],axis=1))
    df_test = df_test[X_train.columns]
    prediction = selected_model.predict(df_test)
    submission_df = pd.DataFrame()
    submission_df['SK_ID_CURR'] = df_test['SK_ID_CURR']
    submission_df['TARGET'] = prediction
    submission_df.to_csv('home_credit_submission.csv', ',', index=False)


def make_subset(df, subset=0):
    if subset > 0:
        df = df[:subset]
    return df




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
    # df_application, df_application_test = featureSelection(df_application, df_application_test)

df = pd.read_pickle('application_processed.pickle')
df_test = pd.read_pickle('application_test_processed.pickle')

run(df, df_test, True)
