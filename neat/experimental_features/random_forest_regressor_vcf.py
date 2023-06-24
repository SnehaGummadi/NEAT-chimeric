# 2/28/22

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

# Take input VCF --> read CSV and create pandas data frame

### Code from https://gist.github.com/dceoy/99d976a2c01e7f0ba1c813778f9db744

def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

H1N1_vcf = read_vcf('/Users/keshavgandhi/Downloads/trio.2010_06.ychr.sites.vcf')

print(H1N1_vcf.columns)

# Data pre-processing - need to one-hot encode REF and ALT columns because random forest does not like strings
### From https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
### Generate binary values using dummy encoding and avoid dummy trap by dropping the first base

ref_bases_df = pd.concat([H1N1_vcf, pd.get_dummies(H1N1_vcf['REF'], prefix='REF_base', drop_first=True)], axis=1)
alt_ref_bases_df = pd.concat([ref_bases_df, pd.get_dummies(H1N1_vcf['ALT'], prefix='ALT_base', drop_first=True)], axis=1)

# Split data into X and y to predict quality scores from other features

### Independent variables - position on chromosome, chromosome number, reference allele, later include trinucleotide
### context

### Use human VCF on Box and chr() for information relating to CHROM

### Quality score is the dependent variable

y = H1N1_vcf['QUAL']

### Dropping quality scores and other nonnumerical information (also dropping REF and ALT because we have dummies)

X = H1N1_vcf.drop(['QUAL', 'ALT', 'REF', 'CHROM', 'FILTER', 'ID', 'INFO'], axis=1)

### May have to vary testing size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Observe shape of data

print('Shape of original dataset:      ', H1N1_vcf.shape)
print('Shape of input - training set:  ', X_train.shape)
print('Shape of output - training set: ', y_train.shape)
print('Shape of input - testing set:   ', X_test.shape)
print('Shape of output - testing set:  ', y_test.shape)

# Convert y_train and y_test to arrays and reshape

y_test_array = y_test.to_numpy()
y_test_array = y_test_array.reshape(-1, 1)
print(y_test_array.shape)

# Define the model

random_state = 0
random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=random_state)

# Fit the regressor using the training data --> make predictions --> score models

random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

y_pred = y_pred.reshape(-1, 1)

print(random_forest.score(y_test_array, y_pred))

# Cross-validation --> need to choose either k folds or grid search

# kfcv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# gscv = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)
# gscv.fit(X_train, y_train)
# print(gscv.best_params_)

# # Tuning

# ### Hyperparameter investigation - number of samples, features, trees, tree depth

# random_forest_tuning = RandomForestRegressor(random_state = SEED)
# param_grid = {
#    'n_estimators': [100, 200, 500],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [4,5,6,7,8],
#    'criterion' :['mse', 'mae']
# }

# # Bootstrapping/bagging

# ?

# # Testing and error metrics

# random_forest = RandomForestRegressor(random_state = SEED)
# random_forest.fit(X_train, y_train)
# y_pred = random_forest.predict(X_test)
# print('MAE: ', mean_absolute_error(y_test, y_pred))
# print('MSE: ', mean_squared_error(y_test, y_pred))

# # Generalizability - out-of-box score

# random_forest_out_of_bag = RandomForestRegressor(oob_score=True)
# random_forest_out_of_bag.fit(X_train, y_train)
# print(random_forest_out_of_bag.oob_score_)
