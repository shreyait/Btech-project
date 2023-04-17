# -*- coding: utf-8 -*-
"""loan defaulter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WHcrnmgD2Qb4hnfmPvaEWdBOhMBuG7DS
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder

df_train=pd.read_csv('/content/train')
df_train.head()

df_train.info()

# df_train['employment_type'] = df_train['employment_type'].map({'Self employed':1,'Salaried':2})

df_train = pd.get_dummies(df_train, columns = ['employment_type'])
df_train.head(2)

df_train.drop(['unique_id','loan_manager_id','mobile_no_available','aadhaar_available','pan_available','voter_id_available','driving_licence_available','passport_available','area_code'],axis = 1,inplace=True)

tc = df_train.corr()
plt.figure(figsize=(18,18))
sns.heatmap(tc, annot = True, cmap ='plasma',
            linecolor ='black', linewidths = 1)

#avg_account_age and credit_history_length are 0.59 correlation so we can remove one
df_train.drop('credit_history_length',axis=1,inplace=True)

df_train.isnull().sum()

df_train.shape

from sklearn.model_selection import train_test_split
X=df_train.drop('loan_defaulted', axis=1,inplace=False)
Y=df_train['loan_defaulted']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=41)

from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)

X_train.drop(['overdue_accounts','no_of_inquiries_in_last_month','no_of_loan_accounts','applicant_age','employment_type_Salaried','employment_type_Self employed'],axis = 1,inplace=True)
X_test.drop(['overdue_accounts','no_of_inquiries_in_last_month','no_of_loan_accounts','applicant_age','employment_type_Salaried','employment_type_Self employed'],axis = 1,inplace=True)

from sklearn.model_selection import train_test_split
# X=df_train.drop('loan_defaulted',axis=1,inplace=False)
# y=df_train['loan_defaulted']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=2, min_samples_split=3,min_samples_leaf=1,max_features='auto',max_depth=3,bootstrap=True)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')

#  from sklearn.model_selection import RandomizedSearchCV
# from pprint import pprint
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 2, stop = 200, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(start=3, stop=10, num = 1)]
# # Minimum number of samples required to split a node
# min_samples_split = [2, 3, 5,7]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}
# pprint(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)

# rf_random.best_params_

# 1. RandomForestClassifier(max_depth=4, max_features='auto', min_samples_leaf=2,
#                        min_samples_split=5, n_estimators=48)
# accuracy : 0.8235039778623314
#  f1 : score 0.7437974468803511
# 2. rf_random.best_params_
# {'n_estimators': 2,
#  'min_samples_split': 3,
#  'min_samples_leaf': 1,
#  'max_features': 'auto',
#  'max_depth': 3}
#  accuracy : 0.8233310273261847
# f1 score : 0.7455087768271014

X_test.head(2)

yhat = rf_model.predict(pd.DataFrame([[47805,74592,77.293384,51,2,0,0,0.004315,0,4376,121300,1899]], columns=['loan_amount','asset_cost','loan_to_asset_value_ratio','asset_manufacturer_id','credit_score','new_loan_accounts_in_last_6_months','overdue_accounts_in_last_6_months','avg_account_age','active_loan_accounts','existing_loan_balance','total_disbursed_amount','current_installment']))

print(yhat)

import pickle

pickle.dump(rf_model, open('usedvehiclehost.pkl', 'wb'))