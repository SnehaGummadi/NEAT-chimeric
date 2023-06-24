# 3/11/22

import io
import os
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

# Goal: use quality scores by position to predict next quality score at that position

# Take input SAM --> lists --> create pandas data frame

file = "/Users/keshavgandhi/Downloads/baby.bam"

file_to_parse = pysam.AlignmentFile(file, check_sq=False)

qual_list = []
unmap_list = []

for item in file_to_parse:

    # For reference
    sam_flag = item.flag
    my_ref = item.reference_id
    mate_ref = item.next_reference_id
    my_tlen = abs(item.template_length)

    # Mapping quality scores
    align_qual = item.query_alignment_qualities

    # Convert list of scores to list of error rates
    align_qual = [10 ** -(i / 10) for i in align_qual]

    # Convert list of error rates to list of success rates (if needed)
    align_qual = [1 - i for i in align_qual]

    # Mapping reference positions
    ref_pos = item.get_reference_positions

    # Mapping lack of paired reads (and binarize)
    unmap_mate = int(item.mate_is_unmapped)

    # Append to master lists
    qual_list.append(align_qual)
    unmap_list.append(unmap_mate)

# Turn list (of lists) into a dataframe
H1N1_sam = pd.DataFrame(qual_list, unmap_list)

print(H1N1_sam)

### Examples

### Convert from error rate to quality score:

# error_rate = 0.0001
# quality_score = int(-10 * np.log10(error_rate))

### Convert quality score back to a character (may not need)

# quality_score = chr(quality_score + 33)
#
# quality = "GGGGGGFGDFFF;GGBGEGG@GGGFGGGEGCG>GFFGFFG>GFGFEGD0GGEGFF@AGGGG=C%GGGGGAGF97<F4GBFEDG=GGEC>6DGGC9EBFFAA"
#
# qual_list = []
# for i in quality:
#     qual_list.append(ord(i) - 33)
#
# phred_score = 38
# 10 ** -(phred_score / 10)

### Error rate

# output: 10 ** -(phred_score / 10)
# [1010 ** -(i / 10) for i in qual_list]

# Get other features from the SAM file

### Use position on the reference as a potential feature --> item.get_reference_positions --> returns list (keep first)
### --> group together by chromosome, also called contig (since that can also be a predictive feature) --> may have to
### parameritize contig --> assign to list

### Use mate being mapped as a feature (reads without a mate could be low quality) - item.mate_is_unmapped --> binarize

# Split data into X and y to predict quality scores from other features

### Dependent variable - start with a single position being predicted by the metrics above

# y = H1N1_sam['']

### Dropping Y columns - need to figure out what columns to use!

# X = H1N1_sam.drop([''], axis=1)

### May have to vary testing size

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Observe shape of data

# print('Shape of original dataset:      ', H1N1_sam.shape)
# print('Shape of input - training set:  ', X_train.shape)
# print('Shape of output - training set: ', y_train.shape)
# print('Shape of input - testing set:   ', X_test.shape)
# print('Shape of output - testing set:  ', y_test.shape)

# Convert y_train and y_test to arrays and reshape

# y_test_array = y_test.to_numpy()
# y_test_array = y_test_array.reshape(-1, 1)
# print(y_test_array.shape)

# Define the model

# random_state = 0
# random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=random_state)

# Fit the regressor using the training data --> make predictions --> score models

# random_forest.fit(X_train, y_train)
# y_pred = random_forest.predict(X_test)
#
# y_pred = y_pred.reshape(-1, 1)
#
# print(random_forest.score(y_test_array, y_pred))

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
