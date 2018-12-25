from ml_utils import make_subset, load_intermediate, save_intermediate
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


def load_data():
    df_application = pd.read_csv('../data/train.csv')
    df_application_test = pd.read_csv('../data/test.csv')
    # df_bureau_balance = pd.read_csv('bureau_balance.csv')
    # df_installment_payments = pd.read_csv('installments_payments.csv')
    # df_bureau = pd.read_csv('bureau.csv')
    # df_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    # df_prev_application = pd.read_csv('previous_application.csv')
    # return df_application,df_application_test, df_bureau, df_bureau_balance, df_cash_balance, df_installment_payments, df_prev_application
    return df_application, df_application_test

def stage_1(subset=0):
    # load data
    global df_application
    global df_application_test


    df_application, df_application_test = load_data()

    df_application = make_subset(df_application, subset)
    df_application_test = make_subset(df_application_test, subset)
    print("{},{}".format(df_application.shape, df_application_test.shape))

    save_intermediate(df_application, df_application_test)
    print(psutil.Process(pid=os.getpid()).memory_full_info())

# Nulls
# Scale
# Transform
# Binn
# Categoricals






# if len(sys.argv) < 2:
#     print("enter step number")
#     exit()
steps = []

count = 0
max_steps =1
for step in sys.argv:
    if count > 0:
        steps.append(int(step))
    count+=1

if steps == []:
    steps = range(1,max_steps+1)

#steps = [1,2,3,4,5,6,7]
for step in steps:
    print("executing step {}".format(step))
    exec("stage_{}()".format(step))

