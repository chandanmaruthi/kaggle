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

# bureau_balance.csv
# bureau.csv
# credit_card_balance.csv
# installments_payments.csv
# POS_CASH_balance.csv
# previous_application.csv
# sample_submission.csv

#df_application = df_application[:1000]
df_application_test = df_application_test[:1000]

df_application.head(5)
df_application.shape

# Create a label encoder object
le = LabelEncoder()
le_count = 0
oh_count = 0
oh_df = pd.DataFrame()
# Iterate through the columns
for col in df_application:
    if df_application[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df_application[col].unique())) <= 2:
            print('le :' + col)
            # Train on the training data
            le.fit(df_application[col])
            # Transform both training and testing data
            df_application[col] = le.transform(df_application[col])
            # df_application[col] = le.transform(df_application[col])

            # Keep track of how many columns were label encoded
            le_count += 1
        else:
            print('ce: ' + col)
            oh_df = pd.concat([oh_df, pd.get_dummies(df_application[col], prefix=col + '_')], axis=1)
            df_application.drop([col], axis=1, inplace=True)

print('%d columns were label encoded.' % le_count)

df_application= pd.concat([df_application,oh_df],axis=1)

df_application.shape
df_application.dropna(axis=1,inplace=True)

def compute_score(clf, X, y, scoring='accuracy'):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    xval = cross_val_score(clf, X, y, cv = 10, scoring=scoring)
    return np.mean(xval)

from sklearn.model_selection import train_test_split
X = df_application.drop(['TARGET'],axis=1)
y = df_application['TARGET']
train_features, test_features, train_labels, test_labels = \
train_test_split(X, y, test_size=0.1, random_state=42)


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
rf_model.fit(train_features, train_labels)


#predictions = rf_model.predict(test_features)
accuracy =  compute_score(rf_model,test_features,test_labels)
print(accuracy)
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, rf_model.predict_proba(test_features)[:,1])
print(auc(false_positive_rate, true_positive_rate))

#roc_auc_score(Y_test, clf.predict(xtest))
