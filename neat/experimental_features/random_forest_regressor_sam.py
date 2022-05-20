# 5/19/22

import io
import os
import pysam
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Goal: use quality scores by position to predict next quality score at that position

# Take input SAM --> lists --> create pandas data frame

bam_file = '/home/joshfactorial/Documents/neat_data/machine_learning_data/sixteenth.bam'

file_to_parse = pysam.AlignmentFile(bam_file, check_sq=False)

print(file_to_parse.count())

modulo = round(file_to_parse.count() / 100) + 1

pos_list = []
qual_list = []
unmap_list = []
i = 0
j = 0
print("Parsing file")
for item in file_to_parse:

    if item.is_unmapped:
        continue
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

    # Mapping reference positions - original position of each nucleotide in the BAM record (deletions would have
    # "skipping" of numbers) --> current base and the base before (combinations of letters) --> assign a number -->
    # encode categorically

    ref_pos = item.get_reference_positions()

    # Mapping lack of paired reads (and binarize)
    unmap_mate = int(item.mate_is_unmapped)

    # Append to master lists
    qual_list.append(align_qual)
    unmap_list.append(unmap_mate)
    pos_list.append(ref_pos)
    i += 1
    if i % modulo == 0:
        j += 10
        print(f'{j}% complete', end='\r')
print(f'100% complete')

# Turn list (of lists) into a dataframe

subsample_sam = pd.DataFrame(qual_list)

# subsample_sam_test = subsample_sam[['average', 'unmap_mate', 50]]
# subsample_sam_test = subsample_sam[['average', 50]]

# subsample_sam = subsample_sam_test
subsample_sam = subsample_sam.fillna(0)

subsample_sam['unmap_mate'] = unmap_mate

subsample_sam['sum'] = subsample_sam.sum(axis=1)
subsample_sam['average'] = subsample_sam['sum'] / 100

nucleotides = ['A', 'C', 'T', 'G']
dinucs = [x + y for x in nucleotides for y in nucleotides]

dinuc_dict = {}
for i in range(16):
    dinuc_dict[dinucs[i]] = i

# Read group

file_to_parse = pysam.AlignmentFile(bam_file, check_sq=False)

output_header = file_to_parse.header
read_groups = output_header['RG']

# Each dictionary is a separate read group - physicality of reads on slide

group_categories = {}

for x, n in zip(read_groups, range(len(read_groups))):
    group_categories.update({x['ID']: n + 1})

categories_by_record = []

for entry in file_to_parse:
    group = entry.get_tag('RG') # get_tag is correct
    current_category = group_categories[group]
    categories_by_record.append(current_category)

# For each BAM record --> assign and categorize by read group (1, 2, 3, or 4) --> throw out "no match" categories and
# then convert to column in data frame

# print(group_categories) # correct
# print(set(categories_by_record)) # correct

read_group_data = pd.DataFrame(categories_by_record, columns=['read_group'])

read_group_data = pd.get_dummies(data=read_group_data, prefix='read_group', columns=['read_group'], drop_first=False)

subsample_sam['read_group'] = read_group_data['read_group_3'] # first column only - hardcoded because of dummy variable
# trap

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

# y = subsample_sam[5]
y = subsample_sam[50]
# y = y.fillna(0)

print(y)

### Dropping Y columns - need to figure out what columns to use!

# X = subsample_sam.drop([5], axis=1)
X = subsample_sam.drop([50], axis=1)
# X = X.fillna(0)

### May have to vary testing size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

# Observe shape of data

print('Shape of original dataset:      ', subsample_sam.shape)
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

# y_pred = y_pred.reshape(65787, 1) # 6.25 hardcoded

y_pred = y_pred.reshape(32861, 1) # 3.125 hardcoded

print(random_forest.score(y_test_array, y_pred))

# Cross-validation --> need to choose either k folds or grid search

# kfcv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_state)
#
# gscv = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
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
