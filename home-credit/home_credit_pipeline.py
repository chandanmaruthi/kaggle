import matplotlib
import numpy as np
import os
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

class util:

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


def apply_transform(df, transforms):
    for col in df:
        if col == 'TARGET':
            continue
        if str(df[col].dtype) not in ['object', 'category']:
            for transform in transforms:
                new_col_name = transform + "_" + col
                #["scale","log_transform","min_max_transform","standard_scale"]
                #for transform in transforms:
                print(new_col_name + "-----------"+transform)
                if transform == 'scale':
                    df[new_col_name] = util.scale_values(df, col)
                elif transform == 'log_transform':
                    df[new_col_name] = util.log_transform_values(df, col)
                elif transform == 'min_max_scale':
                    df[new_col_name] = util.min_max_scale_values(df, col)
                elif transform == 'standard_scale':
                    df[new_col_name] = util.standard_scale_values(df, col)
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

    #df = util.handle_categoricals_target_encode(df,)
    df = util.handle_categoricals(df, df_2)

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

def make_subset(df, subset=0):
    if subset >0:
        df = df[:subset]
    return df

df_application = None
df_application_test = None
arr_in_process_cols=[]
arr_in_process_cols_test=[]

output = mp.Queue()








def prep_1(subset=0):
    # load data
    global df_application
    global df_application_test


    df_application, df_application_test = load_data()

    df_application = make_subset(df_application, subset)
    df_application_test = make_subset(df_application_test, subset)
    print("{},{}".format(df_application.shape, df_application_test.shape))

    save_intermediate(df_application, df_application_test)
    print(psutil.Process(pid=os.getpid()).memory_full_info())

def prep_2():
    df_application, df_application_test = load_intermediate()
    # Clean up data
    df_application = clean_up(df_application)
    df_application_test =  clean_up(df_application_test)
    print("{},{}".format(df_application.shape, df_application_test.shape))
    save_intermediate(df_application,df_application_test)

def prep_3():
    df_application, df_application_test = load_intermediate()
    print(psutil.Process(pid=os.getpid()).memory_full_info())
    df_application = apply_transform(df_application,["scale","log_transform","min_max_transform","standard_scale"])
    df_application_test = apply_transform(df_application_test, ["scale", "log_transform", "min_max_transform", "standard_scale"])
    save_intermediate(df_application, df_application_test)

def prep_4():
    df_application, df_application_test = load_intermediate()
    print(psutil.Process(pid=os.getpid()).memory_full_info())
    df_application, df_application_test = util.handle_categoricals_target_encode(df_application, df_application_test)
    save_intermediate(df_application, df_application_test)

def prep_5():
    df_application, df_application_test = load_intermediate()
    print(psutil.Process(pid=os.getpid()).memory_full_info())
    # # Feature Engineer
    df_application = fe(df_application, df_application_test)
    df_application_test = fe(df_application_test, df_application)
    print("{},{}".format(df_application.shape, df_application_test.shape))
    # print(df_application['TARGET'].unique())
    save_intermediate(df_application, df_application_test)



    # Define an output queue
def prep_6():
    global df_application
    global df_application_test
    global b
    global b_test
    df_application, df_application_test = load_intermediate()
    df_installments_payments = pd.read_csv('./data/installments_payments.csv')
    df_bureau = pd.read_csv('./data/bureau.csv')
    df_pos_cash_balance = pd.read_csv('./data/POS_CASH_balance.csv')
    df_prev_application = pd.read_csv('./data/previous_application.csv')
    df_credit_card_balance = pd.read_csv('./data/credit_card_balance.csv')

    #df_application['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    df_bureau['DAYS_CREDIT_ENDDATE'][df_bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    df_bureau['DAYS_CREDIT_UPDATE'][df_bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    df_bureau['DAYS_ENDDATE_FACT'][df_bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

    df_credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'][df_credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    df_credit_card_balance['AMT_DRAWINGS_CURRENT'][df_credit_card_balance['AMT_DRAWINGS_CURRENT'] < 0] = np.nan


    get_agg_numerics_data(df_application, 'application', df_application,['SK_ID_CURR'], output,True)
    get_agg_numerics_data(df_application_test, 'application', df_application_test, ['SK_ID_CURR'], output,False)
    get_agg_numerics_data(df_application,'installments_payments', df_installments_payments,['SK_ID_CURR'], output,True)
    get_agg_numerics_data(df_application_test,'installments_payments', df_installments_payments, ['SK_ID_CURR'], output,False)
    get_agg_numerics_data(df_application,'bureau', df_bureau,['SK_ID_CURR'], output,True)
    get_agg_numerics_data(df_application_test, 'bureau', df_bureau, ['SK_ID_CURR'], output,False)
    get_agg_numerics_data(df_application,'POS_CASH_balance', df_pos_cash_balance, ['SK_ID_CURR'], output,True)
    get_agg_numerics_data(df_application_test,'POS_CASH_balance', df_pos_cash_balance, ['SK_ID_CURR'], output,False)
    get_agg_numerics_data(df_application,'previous_application', df_prev_application, ['SK_ID_CURR'], output,True)
    get_agg_numerics_data(df_application_test,'previous_application', df_prev_application, ['SK_ID_CURR'],output,False)
    #get_agg_numerics_data, 'args1':[df_application,'POS_CASH_balance', df_credit_card_balance, ['SK_ID_CURR'],output,True]})
    #parallel_funcs.append({'id':12, 'function1': get_agg_numerics_data, 'args1':[df_application_test,'POS_CASH_balance', df_credit_card_balance, ['SK_ID_CURR'],output,False]})

    arr_in_process_cols = []
    arr_in_process_cols_test = []
    arr_in_process_cols.append(df_application)
    arr_in_process_cols_test.append(df_application_test)

    print("starting multi process of {}".format(len(parallel_funcs)))
    #arr_in_process_cols_tmp, arr_in_process_cols_test_tmp = runInParallel(parallel_funcs,False)
    import time
    n_batch =10
    #parallel_funcs_2 = parallel_funcs_2[:10]
    epochs = len(parallel_funcs_2)//n_batch
    print(psutil.Process(pid=os.getpid()).memory_full_info())
    for epoch in range(epochs):
        print("running epoch {} of {}".format(epoch, epochs))
        arr_in_process_cols_tmp, arr_in_process_cols_test_tmp = runInParallel(parallel_funcs_2[epoch*n_batch:(epoch*n_batch)+n_batch], True)
        b=[]
        b_test=[]
        print(len(arr_in_process_cols_tmp))
        print(len(arr_in_process_cols_test_tmp))

        for df_tmp in arr_in_process_cols_tmp:
            df_application = pd.concat([df_application, df_tmp],axis=1)
        df_tmp = None
        for df_tmp in arr_in_process_cols_test_tmp:
            df_application_test = pd.concat([df_application_test, df_tmp],axis=1)

        print("{},{}".format(df_application.shape, df_application_test.shape))
        time.sleep(5)

    print("check for duplicate cols")
    df_application= duplicate_columns(df_application)
    df_application_test= duplicate_columns(df_application_test)

    df_application = df_application.reset_index()
    df_application_test = df_application_test.reset_index()
    df_application = df_application.replace([np.inf, -np.inf], 0)
    df_application_test = df_application_test.replace([np.inf, -np.inf], 0)

    print("aligning datasets")
    # Align Datasets
    df_application, df_application_test = align_datasets(df_application,df_application_test)

    print("{},{}".format(df_application.shape, df_application_test.shape))

    #print(Counter(df_application))
    # Checkpoint Save
    df_application.to_pickle('./data/application_processed.pickle')
    df_application_test.to_pickle('./data/application_test_processed.pickle')
    print("{},{}".format(df_application.shape, df_application_test.shape))


def duplicate_columns(frame):
    arr_cols = []
    arr_col_ind =[]
    df= pd.DataFrame()
    for id in range(len(frame.columns)-1):
        if frame.iloc[:, id].name not in arr_cols:
            #df= pd.concat([df, frame.iloc[:,id]], axis=1)
            arr_col_ind.append(id)
            arr_cols.append(frame.iloc[:, id].name)
    print(arr_col_ind)
    df = pd.concat([df, frame.iloc[:, arr_col_ind]], axis=1)
    return df
b=[]
b_test = []
def totuple(a):
    try:
        return tuple(i for i in a)
    except TypeError:
        return a

def log_result(results):
    global b
    global b_test
    print('in log results')
    for result in results:
        a = result[0]
        a_test = result[1]
        print('logging multiprocess result: {} {}'.format(a.shape if a is not None else 0 , a_test.shape if a_test is not None else 0))
        b.append(a)
        b_test.append(a_test)

def log_error(error):
    print(error)
multiprocessing_args = []
def runInParallel(fns, final=False):
    global multiprocessing_args
    global b
    global b_test
    proc = []

    function = fns[0]['function1']
    for row in fns:
        multiprocessing_args.append(row['args1'])
    args = range(len(multiprocessing_args))
    pool = multiprocessing.Pool(processes=3)
    results =pool.map_async(agg_single, args, callback = log_result, error_callback=log_error).wait()
    pool.close()
    pool.join()

    return b, b_test



def get_agg_numerics_data(df, file_name,  additional_data, exclude_cols, output, train= True):

    # print('a1')

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
    # print('a2')
    for groupby_cols, specs in AGGREGATION_RECIPIES:
        group_object = additional_data.groupby(groupby_cols)
        for select, agg in specs:
            counter +=1
            groupby_aggregate_name = '{}_{}_{}_{}_{}'.format(str(counter),file_name, '_'.join(groupby_cols), agg, select)
            # print("adding process {}".format(groupby_aggregate_name))
            parallel_funcs_2.append({'id': 10+counter*0.001, 'function1': agg_single,
                                   'args1': [train, select, agg, group_object, groupby_aggregate_name, groupby_cols]})
    # print('a3')

def agg_single(index):
    train, select, agg, group_object, groupby_aggregate_name, groupby_cols = multiprocessing_args[index]
    # global df_application
    # global df_application_test
    print('single processing {}'.format(groupby_aggregate_name))
    df_application , df_application_test = load_intermediate()
    if train:
        df = df_application
    else:
        df = df_application_test
    if df is None:
        print("df is none")
        exit()
    arr_in_process_cols=None
    arr_in_process_cols_test=None
    print("pre merge")
    df = df.merge(group_object[select]
                  .agg(agg)
                  .reset_index()
                  .rename(index=str,
                          columns={select: groupby_aggregate_name})
                  [groupby_cols + [groupby_aggregate_name]],
                  on=groupby_cols,
                  how='inner')
    print("post merge")
    if train:
        arr_in_process_cols= df[groupby_aggregate_name]
    else:
        arr_in_process_cols_test = df[groupby_aggregate_name]
    print('completed single proc')
    return arr_in_process_cols,arr_in_process_cols_test

def load_data():
    df_application = pd.read_csv('./data/application_train.csv')
    df_application_test = pd.read_csv('./data/application_test.csv')
    # df_bureau_balance = pd.read_csv('bureau_balance.csv')
    # df_installment_payments = pd.read_csv('installments_payments.csv')
    # df_bureau = pd.read_csv('bureau.csv')
    # df_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    # df_prev_application = pd.read_csv('previous_application.csv')
    # return df_application,df_application_test, df_bureau, df_bureau_balance, df_cash_balance, df_installment_payments, df_prev_application
    return df_application, df_application_test
    # Feature Selection
    #df_application, df_application_test = featureSelection(df_application, df_application_test)


def load_intermediate():
    df_application = pd.read_pickle('./data/df_train.pickle')
    df_application_test = pd.read_pickle('./data/df_test.pickle')
    return df_application, df_application_test

def save_intermediate(df_application, df_application_test):
    df_application.to_pickle('./data/df_train.pickle')
    df_application_test.to_pickle('./data/df_test.pickle')


#prep_1()
#prep_2()
#prep_3()
#prep_4()
#prep_5()
prep_6()


