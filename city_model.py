### Churn Case Study ###

import pandas as pd
import numpy as np
import models
import features

# Create a dataframe 
df = pd.read_csv('Data/churn.csv')
df.head()

# create 3 separate df - one for each city to see if feature importance differs by city
df_k = df[df['city'] == 'King\'s Landing']
df_k.head()

df_w = df[df['city'] == 'Winterfell']
df_w.head()

df_a = df[df['city'] == 'Astapor']
df_a.head()

## pipeline for converting to categorical ##

# clean the subsets
ordinal = {'luxury_car_user':{False:0,True:1}}
categorical = ['city','phone']
date_manip = ['signup_date']
cont = ['avg_dist','avg_rating_by_driver','avg_rating_of_driver','avg_surge','surge_pct','trips_in_first_30_days','weekday_pct']

# try with df_k
models = reload(models)
features = reload(features)
feat, X = features.get_basic_features(df_k, ordinal, categorical, date_manip, cont)

y = df_k['active'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print "logit_k"
models.run_logit(X_train, X_test, y_train, y_test)
print "knn_k"
models.run_knn(X_train, X_test, y_train, y_test)
print "random forest_k"
models.run_randomforestclassifier(X_train, X_test, y_train, y_test)
print "adaboost_k"
models.run_adaboost(X_train, X_test, y_train, y_test)

# try with df_a
models = reload(models)
features = reload(features)
feat, X = features.get_basic_features(df_a, ordinal, categorical,date_manip ,cont)

y = df_a['active'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print "logit_a"
models.run_logit(X_train, X_test, y_train, y_test)
print "knn_a"
models.run_knn(X_train, X_test, y_train, y_test)
print "random forest_a"
models.run_randomforestclassifier(X_train, X_test, y_train, y_test)
print "adaboost_a"
models.run_adaboost(X_train, X_test, y_train, y_test)

# try with df_w
models = reload(models)
features = reload(features)
feat, X = features.get_basic_features(df_w, ordinal, categorical,date_manip ,cont)

y = df_w['active'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print "logit_w"
models.run_logit(X_train, X_test, y_train, y_test)
print "knn_w"
models.run_knn(X_train, X_test, y_train, y_test)
print "random forest_w"
models.run_randomforestclassifier(X_train, X_test, y_train, y_test)
print "adaboost_w"
models.run_adaboost(X_train, X_test, y_train, y_test)
