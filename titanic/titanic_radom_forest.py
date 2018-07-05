# Lets import some libraries we will use
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
import numpy as np
import os
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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


def get_data_try_1():
    print(pretty_line_sep)
    print("Step 1")
    print(">> Here we will work with data, \nLets loads some data and see what we have to work with")
    print(">> Loading training data")
    train_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/train.csv'))
    print("\n\n>> Lets see the first 10 lines of this data set")
    print(train_data.head(10))
    print(">> Lets see some stats about this data\n\n")
    print(">> Note: pandas describe provides stats on numeric data only so, you dont  see all columns here\n\n")
    print(train_data.describe())
    print(pretty_line_sep)


    print(">> sklearn classifiers require numeric data, so lets start by just dropping test columns")
    train_data_test_1 = train_data.drop(['Name','Ticket', 'Sex','Embarked','Cabin'], axis=1)

    print(">> sklearn classifiers do not like nulls so lets fill in nulls")

    train_data_test_1['Age'].fillna(train_data.Age.mean(), inplace=True)

    print(">> sklearn classifiers require numeric data so we will replace categoricals with numbers")

    # Using Skicit-learn to split data into training and testing sets


    print(train_data.Cabin.unique())




    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = \
                train_test_split(train_data_test_1.drop(['Survived'], axis=1), train_data.Survived, test_size = 0.1, random_state = 42)

    #print(train_features.head(20))
    print(pretty_line_sep)
    return train_features, test_features, train_labels, test_labels


def encoder_fit(id,type,data, target=None):
    global global_encoders
    if type == "ONE_HOT":
        enc = ce.OneHotEncoder()
        enc.fit(data)

    if type == "TARGET":
        enc = ce.TargetEncoder()
        enc.fit(data,target)

    global_encoders.append({"id": id, "encoder": enc})


def encoder_transform(id,data,target=None):
    global global_encoders
    for enc_element in global_encoders:
        if enc_element["id"]==id:
            if target != None:
                trns_data = enc_element["encoder"].transform(data)
            else:
                trns_data = enc_element["encoder"].transform(data,target)
            trns_data.columns = [col_name+'_{}'.format(id) for col_name in trns_data.columns]
            return trns_data


def get_data_try_2():
    train_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/train.csv'))
    train_data['Age'].fillna(train_data.Age.mean(), inplace=True)
    train_data.Sex = pd.get_dummies(train_data.Sex)
    train_data.Cabin = train_data['Cabin'].fillna(value='UnSpecfied')
    train_data['Embarked']= pd.get_dummies( train_data.Embarked)

    #===================================================================
    passengers_names_train = train_data.Name
    test_data_full = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/test.csv'))
    passenger_names_test = test_data_full.Name
    passengers_legend_train =[]
    for name in passengers_names_train:
        passengers_legend_train.append(name.split(" ")[1].strip())

    for name in passenger_names_test:
        passengers_legend_test.append(name.split(" ")[1].strip())

    passengers_legend = passengers_legend_test +passengers_legend_train
    encoder_fit("passenger_legend","TARGET",passengers_legend,train_data.Survived.values)
    passengers_legend_transformed_cols = encoder_transform("passenger_lengend",passengers_legend_train)
    #===============================================================
    #print(train_data.describe())

    #enc_cabin.fit(train_data.Cabin.values)
    #enc_ticket.fit(train_data.Ticket.values)
    all_cabins = train_data.Cabin+ test_data_full.Cabin
    all_tickets = train_data.Ticket + test_data_full.Ticket
    #print(all_cabins)

    encoder_fit("cabin","ONE_HOT",all_cabins.values)
    encoder_fit("ticket","ONE_HOT",all_tickets.values)
    tickets_transformed_cols = encoder_transform("ticket",train_data.Ticket.values)
    cabin_transformed_cols = encoder_transform("cabin",train_data.Cabin.values)
    train_data.drop(['Name','Cabin','Ticket'], axis=1,inplace=True)

    train_data = pd.concat([train_data,passengers_legend_transformed_cols,cabin_transformed_cols, tickets_transformed_cols],axis=1)
    train_features, test_features, train_labels, test_labels = \
                train_test_split(train_data.drop('Survived', axis=1), train_data.Survived, test_size = 0.1, random_state = 42)
    return train_features, test_features, train_labels, test_labels











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
    accuracy = roc_auc_score(test_labels, predictions)
    record_model("Test 1",rf_model,accuracy)




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
    print(features_importance_df.sort_values('Importances',ascending=False))

    predictions = rf_model.predict(test_features)
    accuracy = roc_auc_score(test_labels, predictions)
    record_model("rf_try_2", rf_model, accuracy)
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
    pred_df = pd.DataFrame(columns=['PassengerId','Survived'])
    pred_df.PassengerId = test_features.PassengerId
    pred_df.Survived = predictions


    accuracy = roc_auc_score(test_labels, predictions)
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
    accuracy = roc_auc_score(test_labels, predictions)
    print(pretty_line_sep)
    print(">>XGB : ROC AUC Score {}".format(accuracy))
    
    record_model("xgb_1", xgb_model, accuracy)

    print(pretty_line_sep)
#=====================================================================
#test
def gen_test_submission():
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/test.csv'))
    test_data['Age'].fillna(test_data.Age.mean(), inplace=True)
    test_data['Fare'].fillna(test_data.Fare.mean(), inplace=True)
    test_data.Sex = pd.get_dummies(test_data.Sex)
    test_data.Cabin = test_data['Cabin'].fillna(value='UnSpecfied')

    test_data.Embarked = pd.get_dummies(test_data.Embarked)
    passengers_legend_cols = encoder_transform("passenger_legend",passengers_legend_test)

    #test_data.Cabin =     cabin_le = LabelEncoder().fit(train_data.Cabin)
    test_data_cabin_cols= encoder_transform('cabin', test_data.Cabin.values)
    test_data_ticket_cols = encoder_transform('ticket',test_data.Ticket.values)
    test_data = pd.concat([test_data,passengers_legend_cols,test_data_cabin_cols,test_data_ticket_cols],axis=1)
    #print(test_data.describe())
    test_data.drop(['Name','Cabin','Ticket'], axis=1,inplace=True)
    predictions = best_model.predict(test_data)
    submission_df = pd.DataFrame(columns=['PassengerId','Survived'])
    submission_df.PassengerId= test_data.PassengerId
    submission_df.Survived = predictions
    submission_df.to_csv(os.path.join(os.path.dirname(__file__),'submission_1.csv'), index=False)

train_features, test_features, train_labels, test_labels =get_data_try_1()
rf_model_1(train_features, test_features, train_labels, test_labels)
train_features, test_features, train_labels, test_labels =get_data_try_2()
model_rf_2(train_features, test_features, train_labels, test_labels)
model_gbm_2(train_features, test_features, train_labels, test_labels)
model_xgb_2(train_features, test_features, train_labels, test_labels)
#gen_test_submission()
for test_result in test_results:
    print("{}: {}".format(test_result["Model_Title"],test_result["Model_Score"]))



