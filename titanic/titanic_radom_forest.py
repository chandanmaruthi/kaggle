# Lets import some libraries we will use
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
import numpy as np
import os
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pandas_summary import DataFrameSummary
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings(action='ignore')

# First lets get some data
# The data sets for this experiment can be found here https://www.kaggle.com/c/titanic
# There are 2 files test.csv and train.csv
pretty_line_sep = "========================================================================================"
test_results =[]
best_model = None
best_score = 0
enc_passenger_legend = ce.OneHotEncoder()
enc_cabin = ce.OneHotEncoder()
enc_ticket = ce.OneHotEncoder()
global_encoders = []
passengers_legend_test = []
kmeans_fare = KMeans()
kmeans_age = KMeans()
select_kbest_model = SelectKBest
selected_features = []

def record_model(model_title, model, score):
    global best_score
    global best_model
    model_info = {}
    model_info["Model_Title"] = model_title
    model_info["Model"] = model
    model_info["Model_Score"] =  score
    if score > best_score:
        print("Adding best model :{} with score {}".format(model_title, score))
        best_model = model
        best_score = score
    test_results.append(model_info)

#
# def get_data_try_1(data_path,train=True):
#     print(pretty_line_sep)
#     print("Step 1")
#     print(">> Here we will work with data, \nLets loads some data and see what we have to work with")
#     print(">> Loading training data")
#     data_df = pd.read_csv(data_path)
#     print("\n\n>> Lets see the first 10 lines of this data set")
#     print(train_data.head(10))
#     print(">> Lets see some stats about this data\n\n")
#     print(">> Note: pandas describe provides stats on numeric data only so, you dont  see all columns here\n\n")
#     #print(train_data.describe())
#     print(pretty_line_sep)
#
#
#     print(">> sklearn classifiers require numeric data, so lets start by just dropping test columns")
#     train_data_test_1 = train_data.drop(['Name','Ticket', 'Sex','Embarked','Cabin'], axis=1)
#
#     print(">> sklearn classifiers do not like nulls so lets fill in nulls")
#
#     train_data_test_1['Age'].fillna(train_data.Age.mean(), inplace=True)
#
#     print(">> sklearn classifiers require numeric data so we will replace categoricals with numbers")
#
#     # Using Skicit-learn to split data into training and testing sets
#
#
#
#
#
#
#     # Split the data into training and testing sets
#     train_features, test_features, train_labels, test_labels = \
#                 train_test_split(train_data_test_1.drop(['Survived'], axis=1), train_data.Survived, test_size = 0.1, random_state = 42)
#
#     #print(train_features.head(20))
#     print(pretty_line_sep)
#     return train_features, test_features, train_labels, test_labels



def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 10, scoring=scoring)
    return np.mean(xval)


def encoder_fit(id,type,data, target=None):
    global global_encoders
    if type == "ONE_HOT":
        enc = ce.OneHotEncoder()
        enc.fit(data)
    if type =="KMEANS":
        enc = KMeans().fit(data)

    if type == "TARGET":
        enc = ce.TargetEncoder()
        enc.fit(data,target)

    global_encoders.append({"id": id, "encoder": enc})


def encoder_transform(id,data,target=None):
    global global_encoders
    for enc_element in global_encoders:
        if enc_element["id"]==id:
            if target == None:
                trns_data = pd.DataFrame(enc_element["encoder"].transform(data))
            else:
                trns_data = pd.DataFrame(enc_element["encoder"].transform(data,target))
            #print(type(trns_data))

            #print(type(trns_data.columns))
            trns_data.columns = [str(col_name)+'_{}'.format(id) for col_name in trns_data.columns]
            return trns_data

def load_data(data_path):
    data_df = pd.read_csv(data_path)
    return data_df


def select_features(X,y,train=True):

    global selected_features

    if train:
        # Create and fit selector
        selector = SelectKBest(f_classif,k=10)
        selector.fit(X, y)
        # Get idxs of columns to keep
        idxs_selected = selector.get_support(indices=True)
        # Create new dataframe with only desired columns, or overwrite existing
        selected_features = list(X.iloc[:,idxs_selected].columns.values)
        return X[selected_features]
    else:
        return X[selected_features]

def preprocess_data(data_path,train=True):
    data_df = load_data(data_path)
    global select_kbest_model
    dfs = DataFrameSummary(data_df)
    # lets look at what data we have here
    print(dfs.columns_stats)


    #===================================================================================
    print(pretty_line_sep)
    print('Missing Values')

    print("We see Cabin has a large % of nulls, lets drop cabin for now")
    data_df.drop(['Cabin'],axis=1, inplace=True)

    print("We see there are 2 missing values for Embarked , lets replace nulls with 'Unspecified' ")
    data_df.Embarked.fillna('Unspecified',inplace=True)

    #data_df.drop(['Age'],axis=1,inplace=True)

    data_df.Age.fillna(data_df.Age.mean(),inplace=True)

    data_df.Fare.fillna(data_df.Fare.mean(),inplace=True)
    #print('Lets Look at column stats again')
    #print(DataFrameSummary(data_df).columns_stats)
    #print('Good we dont have any missing values')

    # ===================================================================================
    print(pretty_line_sep)
    print("Now lets handle Categorical values , we see Name, Sex, Ticket, Embarked are Categoricals")
    #print(data_df.head(5))


    sex_cols = pd.get_dummies(data_df.Sex, prefix="Sex_")
    data_df= pd.concat([data_df,sex_cols],axis=1)
    embarked_cols = pd.get_dummies(data_df.Embarked, prefix="Embarked_")
    data_df= pd.concat([data_df,embarked_cols],axis=1)
    data_df.drop(['Embarked'],axis=1, inplace=True)



    passengers_names_train = data_df.Name
    test_data_full = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/test.csv'))
    passenger_names_test = test_data_full.Name
    passengers_legend_train = []
    for name in passengers_names_train:
        passengers_legend_train.append(name.split(" ")[1].strip())

    for name in passenger_names_test:
        passengers_legend_test.append(name.split(" ")[1].strip())

    passengers_legend = passengers_legend_test + passengers_legend_train
    titles = set(passengers_legend)
    titles_good = ['Mrs.','Rev.','Mlle.','Don.','Master.','Major.','Mr.','Ms.','Mme.','Dr.', 'Col.','Capt.','Miss.','Unspecified']
    passengers_legend = titles_good

    # ===================================================================================










    #data_df["Sex_Pclass_Embarked"] = data_df.Sex.astype(str).str.cat(data_df.Pclass.astype(str))
    # data_df["Sex_Pclass_Embarked"] = data_df["Sex_Pclass_Embarked"].astype(str).str.cat(
    #     data_df.Embarked.astype(str))
    # ef_sex_pclass_embarked = pd.get_dummies(data_df['Sex_Pclass_Embarked'])
    # print(ef_sex_pclass_embarked)
    # print(data_df['Sex_Pclass_Embarked'])
    # print(data_df.head(10))

    # ===================================================================

    #data_df.Fare.fillna(data_df.Fare.mean(), inplace=True)
    #data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean())
    if train:
        encoder_fit("passenger_legend", "ONE_HOT", passengers_legend)
        encoder_fit("fare", "KMEANS",data_df.Fare.values.reshape(-1, 1))
        encoder_fit("age", "KMEANS", data_df.Age.values.reshape(-1, 1))

    fare_kmeans_cols = pd.DataFrame(encoder_transform("fare",data_df.Fare.values.reshape(-1, 1))) #pd.DataFrame(kmeans_fare.predict(data_df.Fare.values.reshape(-1, 1)))
    age_kmeans_cols =  pd.DataFrame(encoder_transform("age",data_df.Age.values.reshape(-1, 1)))  #pd.DataFrame(kmeans_age.predict(data_df.Age.values.reshape(-1, 1)))
    #print(type(fare_kmeans))
    passengers_legend_transformed_cols = encoder_transform("passenger_legend", [x if x in passengers_legend else 'Unspecified' for x in passengers_legend_train])
    data_df = pd.concat([data_df,passengers_legend_transformed_cols,fare_kmeans_cols,age_kmeans_cols],axis=1)
    #print(data_df.head(5))
    #exit()
    # ===============================================================
    # print(data_df.describe())

    # enc_cabin.fit(data_df.Cabin.values)
    # enc_ticket.fit(data_df.Ticket.values)
    # data_df.Cabin.fillna("Unspecified", inplace =True)
    # data_df.Ticket.fillna("UnSpecified",inplace=True)
    # all_cabins = data_df.Cabin + test_data_full.Cabin
    # all_tickets = data_df.Ticket + test_data_full.Ticket
    # print(all_cabins)

    # encoder_fit("cabin","ONE_HOT",all_cabins.values)
    # encoder_fit("ticket","ONE_HOT",all_tickets.values)
    # tickets_transformed_cols = encoder_transform("ticket",data_df.Ticket.values)
    # cabin_transformed_cols = encoder_transform("cabin",data_df.Cabin.values)

    # data_df['Age']= data_df.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    # print("nulls in age? {}".format(data_df.Age.isnull().any()))
    # exit()

    # data_df['Fare'] = fare_kmeans
    # data_df['Age']= age_kmeans
    data_df.drop(['Name', 'Ticket', 'Sex', 'Fare'], axis=1,
                    inplace=True)
    #print(len(data_df.columns))
    print(len(passengers_legend_transformed_cols.columns))
    # print(len(cabin_transformed_cols.columns))
    # print(len(tickets_transformed_cols.columns))
    #data_df = pd.concat([data_df, sex_cols, fare_kmeans.values, age_kmeans], axis=1)
    #print(data_df)
    # print(data_df.describe())
    # exit()


    #====================================================================================
    #Feature Selection

    #if train:
    #    select_kbest_model = SelectKBest(score_func=chi2)
    #    select_kbest_model.fit(data_df.drop('Survived', axis=1), data_df.Survived)

    
    #selected_features = pd.DataFrame(select_kbest_model.transform(data_df.drop('Survived', axis=1) if 'Survived' in data_df.columns else data_df))
    #print(selected_features)
    #exit()
    #====================================================================================







    if train:
        X = data_df.drop('Survived', axis=1)
        y = data_df.Survived
        selected_X = select_features(X,y, True)
        train_features, test_features, train_labels, test_labels = \
            train_test_split(selected_X, y, test_size=0.1, random_state=42)
    else:
        X = data_df
        y = None
        selected_X = select_features(X, y, False)
        train_features = selected_X
        test_features = None
        train_labels = None
        test_labels = None

    # print(train_features.shape)
    # print(train_features.head())
    # print(train_features.describe())

    #exit()
    dfs = DataFrameSummary(train_features)
    # print(pretty_line_sep)
    # print("head")
    # print(data_df.head(10))
    # print(dfs.columns_stats)
    # print(pretty_line_sep)

    return train_features, test_features, train_labels, test_labels


# def linear_model(train_features, test_features, train_labels, test_labels):
#     from sklearn import linear_model
#     l_model = linear_model()


def rf_model_1(train_features, test_features, train_labels, test_labels):
    # Import the model we are using

    from sklearn import tree
    from sklearn.tree import export_graphviz

    # Instantiate model with 1000 decision trees
    rf_model = RandomForestClassifier(n_estimators = 100,
                                      random_state = 42,
                                      n_jobs=-1,
                                      oob_score=True,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features='auto',
                                      bootstrap=True)
    # Train the model on training data
    rf_model.fit(train_features, train_labels)


    predictions = rf_model.predict(test_features)
    accuracy =  compute_score(rf_model,test_features,test_labels)

    record_model("rf_1",rf_model,accuracy)

    importances = rf_model.feature_importances_
    features_importance_df = pd.DataFrame({'Features':list(train_features.columns), 'Importances':importances})



def model_rf_2(train_features, test_features, train_labels, test_labels):
    rf_model = RandomForestClassifier(n_estimators = 1000,
                                      random_state = 42,
                                      n_jobs=-1,
                                      oob_score=True,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features='auto',
                                      bootstrap=True)

    rf_model.fit(train_features, train_labels)

    importances = rf_model.feature_importances_
    features_importance_df = pd.DataFrame({'Features':list(train_features.columns), 'Importances':importances})
    #print(features_importance_df.sort_values('Importances',ascending=False))

    predictions = rf_model.predict(test_features)
    accuracy = compute_score(rf_model, test_features, test_labels)
    record_model("rf_2", rf_model, accuracy)


def model_rf_3(train_features, test_features, train_labels, test_labels):
    from scipy.stats import randint as sp_randint
    from sklearn.model_selection import GridSearchCV
    param_dist = {"max_depth": [3, 5,10,15],
                  "max_features": [1,5,10],
                  "min_samples_split": [2,4,6,8,10,12],
                  "min_samples_leaf": [1,5,10,100],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    model = RandomForestClassifier(n_estimators=20)
    random_search = GridSearchCV(model, param_grid=param_dist,
                                       n_jobs=n_iter_search)
    random_search.fit(train_features, train_labels)

    #predictions = random_search.predict(test_features)
    accuracy = compute_score(random_search, test_features, test_labels)
    record_model("rf__gserach_3", random_search, accuracy)
    print('rf_gsearch_3 score {}'.format(accuracy))
#===========================================================================================

# param_grid = { "criterion" : ["gini", "entropy",],
#                "min_samples_leaf" : [1, 5, 10,100],
#                "min_samples_split" : [2, 4, 10, 12],
#                "n_estimators": [50, 100, 400, 700]}
#
# gs = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# gs = gs.fit(train_features,train_labels)
# print(gs.best_score_)
# print(gs.best_params_)



#============================================================================
def model_gbm_2(train_features, test_features, train_labels, test_labels):
    from sklearn.ensemble import GradientBoostingClassifier
    param_test1 = {'n_estimators':range(20,81,10)}

    gb = GradientBoostingClassifier(criterion='mse',
                  learning_rate=0.01,
                  max_depth=10,
                  n_estimators=100,
                  random_state=42,
                  max_features='sqrt')
    gb.fit(train_features,train_labels)


    predictions = gb.predict(test_features)
    accuracy = compute_score(gb, test_features, test_labels)
    record_model("gbm_1", gb, accuracy)

#======================================================================



#===========================================================================
def model_xgb_2(train_features, test_features, train_labels, test_labels):
    import xgboost as xgb
    xgb_model_def = xgb.XGBClassifier(max_depth=10,
                                      n_estimators=100,
                                      learning_rate=0.01,
                                      random_state=42,
                                      max_features='sqrt')
    xgb_model = xgb_model_def.fit(train_features,train_labels)
    predictions = xgb_model.predict(test_features)
    accuracy = compute_score(xgb_model, test_features, test_labels)
    record_model("xgb_1", xgb_model, accuracy)
#========================================================================

#ensemble
def model_ensemble(train_features, test_features, train_labels, test_labels):

    test_df = pd.read_csv(path)
    predictions = best_model.predict(test_features)
    submission_df = pd.DataFrame(columns=['PassengerId','Survived'])
    print(test_features.head(10))
    submission_df.PassengerId= test_df.PassengerId
    submission_df.Survived = predictions
    submission_df.to_csv(os.path.join(os.path.dirname(__file__),'submission.csv'), index=False)
#========================================================================
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import  BaggingClassifier
SEED=42
def get_models(train_features, train_labels):
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }
    return models

def model_ensemble_1(models,X, y):
    # bag = BaggingClassifier(
    #                         RandomForestClassifier(n_neighbors=3),
    #                         max_samples=0.5,
    #                         max_features=2,
    #                         n_jobs=2,
    #                         oob_score=True)
    rf = RandomForestClassifier(n_estimators=1000,oob_score=True)
    rf.fit(X,y)
    print("ensemble best score : {}".format(rf.oob_score_))

#=====================================================================
#test
def gen_test_submission(path, test_features):
    test_df = pd.read_csv(path)
    predictions = best_model.predict(test_features)
    submission_df = pd.DataFrame(columns=['PassengerId','Survived'])
    print(test_features.head(10))
    submission_df.PassengerId= test_df.PassengerId
    submission_df.Survived = predictions
    submission_df.to_csv(os.path.join(os.path.dirname(__file__),'submission.csv'), index=False)




#train_features, test_features, train_labels, test_labels =preprocess_data(os.path.join(os.path.dirname(__file__),'data/train.csv'),True)
#rf_model_1(train_features, test_features, train_labels, test_labels)
train_features, test_features, train_labels, test_labels =preprocess_data(os.path.join(os.path.dirname(__file__),'data/train.csv'),True)
model_rf_2(train_features, test_features, train_labels, test_labels)
model_gbm_2(train_features, test_features, train_labels, test_labels)
model_xgb_2(train_features, test_features, train_labels, test_labels)
model_rf_3(train_features, test_features, train_labels, test_labels)
models = get_models(train_features, test_labels)
model_ensemble_1(models, train_features,train_labels)
train_features, test_features, train_labels, test_labels =preprocess_data(os.path.join(os.path.dirname(__file__),'data/test.csv'),False)
#print(len(train_features.columns))

gen_test_submission(os.path.join(os.path.dirname(__file__),'data/test.csv'),train_features)
for test_result in test_results:
    print("{}: {}".format(test_result["Model_Title"],test_result["Model_Score"]))



