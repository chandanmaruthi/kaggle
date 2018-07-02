# Lets import some libraries we will use
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
import numpy as np
import os
pd.set_option('display.max_columns', None)

# First lets get some data
# The data sets for this experiment can be found here https://www.kaggle.com/c/titanic
# There are 2 files test.csv and train.csv
print("--------------------------Step 1 -----------------------------------")
print(">> Here we will work with data, \nLets loads some data and see what we have to work with")
print(">> Loading training data")
train_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/train.csv'))
print("\n\n>> Lets see the first 10 lines of this data set")
print(train_data.head(10))
print(">> Lets see some stats about this data\n\n")
print(">> pandas describe provides stats on numeric data only so, you dont  see all columns here\n\n")
print(train_data.describe())

test_results ={}

print(">> sklearn classifiers require numeric data, so lets start by just dropping test columns")
train_data = train_data.drop(['Name','Ticket'], axis=1)

print(">> sklearn classifiers do not like nulls so lets fill in nulls")
train_data.Cabin = train_data['Cabin'].fillna(value='UnSpecfied')
train_data['Age'].fillna(train_data.Age.mean(), inplace=True)

print(">> sklearn classifiers require numeric data so we will replace categoricals with numbers")
train_data.Sex = pd.get_dummies(train_data.Sex)
train_data.Embarked = pd.get_dummies(train_data.Embarked)
train_data.Cabin = pd.get_dummies(train_data.Cabin)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_data_features = train_data.drop(['Survived'], axis=1)
train_features, test_features, train_labels, test_labels = \
            train_test_split(train_data_features, train_data.Survived, test_size = 0.5, random_state = 42)

print(train_features.head(20))
print('>> Training Features Shape:', train_features.shape)
print('>> Training Labels Shape:', train_labels.shape)
print('>> Testing Features Shape:', test_features.shape)
print('>> Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
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
test_results["Test 1"] = accuracy
print(">> We compute a benchmark score of {}".format(accuracy))
print(">> lets see if we can better this")



importances = rf_model.feature_importances_
features_importance_df = pd.DataFrame({'Features':list(train_data_features.columns), 'Importances':importances})
print(features_importance_df.sort_values('Importances',ascending=False))



train_data.drop(['Embarked', 'Cabin','Parch'], axis=1,inplace=True)
train_data_features = train_data.drop(['Survived'], axis=1)

train_features, test_features, train_labels, test_labels = \
            train_test_split(train_data_features, train_data.Survived, test_size = 0.5, random_state = 42)
print(train_features.head(5))
rf_model = RandomForestClassifier(n_estimators = 100,
                                  random_state = 42,
                                  n_jobs=-1,
                                  oob_score=True,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  max_features='auto',
                                  bootstrap=True)
rf_model.fit(train_features, train_labels)


predictions = rf_model.predict(test_features)
accuracy = roc_auc_score(test_labels, predictions)
print(">> ROC AUC Score {}".format(accuracy))
test_results["Test 2"] = accuracy
print(">> lets see if we can better this")



print(test_results)


exit()






















# Use numpy to convert to arrays
# Labels are the values we want to predict
labels = np.array(train_data['Survived'])
# Remove the labels from the features
# axis 1 refers to the columns
train_data.Age = train_data["Age"].fillna(train_data.Age.mean(), inplace=True)
train_data.Sex =  pd.get_dummies(train_data.Sex)
train_data.Embarked =  pd.get_dummies(train_data.Embarked)
train_data = train_data.drop('SibSp', axis=1)
train_data = train_data.drop('Name', axis = 1)
train_data = train_data.drop('Ticket', axis=1)
#features= features.drop('Pclass', axis = 1)
#features= features.drop('Cabin', axis = 1)
#features= features.drop('Parch', axis = 1)
#features= features.drop('SibSp', axis = 1)
#features= features.drop('Embarked', axis = 1)
# Saving feature names for later use
feature_list = list(train_data.columns)
# Convert to numpy array
features = np.array(train_data)




print("\n\n\n\n")

# The baseline predictions are the historical averages
#baseline_preds = test_features[:, feature_list.index('Survived')]
# Baseline errors, and display average baseline error
#baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

test_df = pd.read_csv("test_new.csv")
#============================================================================
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42,max_features="sqrt")
# Train the model on training data
rf.fit(train_features, train_labels)
ar = feature_list
values = sorted(zip(ar, rf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, ar, tablefmt="plain"))



int_count=0
# for tree in rf.estimators_:
#     int_count+=1
#     export_graphviz(tree,
#                     feature_names=feature_list,
#                     filled=True,
#                     rounded=True,out_file="tree_{}.dot".format(int_count))
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)
print("Random Forest : "+str(accuracy_score(test_labels,predictions)))
predictions = rf.predict(test_df)
pred_df = pd.DataFrame(columns=['PassengerId','Survived'])
pred_df.PassengerId = test_df.PassengerId
pred_df.Survived = predictions
pred_df.to_csv("rf_1.csv",index=False)

#===========================================================================
from sklearn.ensemble import GradientBoostingClassifier
param_test1 = {'n_estimators':range(20,81,10)}

# gs = GridSearchCV(estimator= GradientBoostingClassifier(n_estimators = 1000,
#                                  learning_rate=0.01,
#                                  max_depth=10,
#                                  max_features='sqrt',
#                                  random_state = 0,
#                                  subsample=0.8,
#                                  min_samples_split=50,
#                                  min_samples_leaf=50),
#                   param_grid=param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# Train the model on training data
#gs.fit(train_features, train_labels)
#print(gs.grid_scores_, gs.best_params_)
#print(gs.best_estimator_)
#print(gs.best_score_)
gb = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.01, loss='deviance', max_depth=10,
              max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=50, min_samples_split=50,
              min_weight_fraction_leaf=0.0, n_estimators=70,
              presort='auto', random_state=0, subsample=0.8, verbose=0,
              warm_start=False)
gb.fit(train_features,train_labels)


# Use the forest's predict method on the test data
predictions = gb.predict(test_df)
pred_df = pd.DataFrame(columns=['PassengerId','Survived'])
pred_df.PassengerId = test_df.PassengerId
pred_df.Survived = predictions
#print(pred_df)
pred_df.to_csv("gbm_result_1.csv",index=False)

#print("Gradient Boosting : " +str(accuracy_score(test_labels,predictions)))

#===========================================================================
def predict_xgb(features,labels=None):
    import xgboost as xgb
    xgb_model_def = xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.05)
    xgb_model = xgb_model_def.fit(train_features,train_labels)
    predictions = xgb_model.predict(features)
    if labels.all():
        print("XGBoost Gradient Boosting : " +str(accuracy_score(labels,predictions)))
    return predictions




pred_df = pd.DataFrame(columns=['PassengerId','Survived'])
pred_df.PassengerId = test_df.PassengerId
predict_xgb(test_features,test_labels)
pred_df.Survived = predict_xgb(test_df.values)
#print(pred_df)
pred_df.to_csv("xgb_result_1.csv",index=False)




