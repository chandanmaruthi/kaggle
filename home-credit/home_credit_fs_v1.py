from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
import pandas as pd

df = pd.read_pickle('application_processed.pickle')
df_test = pd.read_pickle('application_test_processed.pickle')
X = df.drop(['TARGET'], axis=1)
y = df['TARGET']

lgbc=LGBMClassifier(n_estimators=5000, learning_rate=0.01, num_leaves=100, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

#embeded_lgb_selector = SelectFromModel(lgbc, threshold='.1*median')
embeded_lgb_selector = SelectFromModel(lgbc, threshold=1e-4)
embeded_lgb_selector.fit(X, y)
embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')


df_test = df_test[embeded_lgb_feature]

embeded_lgb_feature.append('TARGET')
df = df[embeded_lgb_feature]
pd.to_pickle(df,'fs_data.pickle')
pd.to_pickle(df_test,'fs_test_data.pickle')
