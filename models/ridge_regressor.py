# 8/8/22

import io
import os
import pysam
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Goal: use quality scores by position to predict next quality score at that position

# Take input SAM --> lists --> create pandas data frame

bam_file = '/Users/keshavgandhi/Downloads/subsample_3.125.bam'

index = f'{bam_file}.bai'

if not pathlib.Path(index).exists():
    print("No index found, creating one.")
    pysam.index(bam_file)

file_to_parse = pysam.AlignmentFile(bam_file, 'rb', check_sq=False)
num_recs = file_to_parse.count()
print(f'{num_recs} records to parse')

modulo = round(num_recs / 9)

pos_list = []
qual_list = []
unmap_list = []
sam_flags = []
tlens = []
skips = []
i = 0
j = 0

def print_update(number, factor, percent):
    if number % factor == 0:
        percent += 10
        print(f'{percent}% complete', end='\r')
    return percent

print("Parsing file")

for item in file_to_parse.fetch():

    if item.is_unmapped:
        i += 1
        j = print_update(i, modulo, j)
        continue
    if len(item.seq) != 249:
        i += 1
        j = print_update(i, modulo, j)
        continue
    if 'S' in item.cigarstring:
        i += 1
        j = print_update(i, modulo, j)
        continue

    # For reference
    sam_flag = item.flag
    sam_flags.append(sam_flag)
    my_ref = item.reference_id
    mate_ref = item.next_reference_id
    my_tlen = abs(item.template_length)
    tlens.append(my_tlen)

    # Mapping quality scores
    align_qual = item.query_alignment_qualities

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
    j = print_update(i, modulo, j)

print(f'100% complete')
file_to_parse.close()

# Turn list (of lists) into a dataframe

subsample_sam = pd.DataFrame(qual_list)

column_names = list(np.arange(0, len(subsample_sam.columns), 1))
column_names = [str(i) for i in column_names]
subsample_sam = subsample_sam.set_axis(column_names, axis='columns', inplace=False)

subsample_sam['unmap_mate'] = unmap_mate

subsample_sam['sum'] = subsample_sam.sum(axis=1)
subsample_sam['average'] = subsample_sam['sum'] / len(subsample_sam.columns)
subsample_sam = subsample_sam.drop(columns=['sum'])

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

read_group_data = pd.DataFrame(categories_by_record, columns=['read_group'])

read_group_data = pd.get_dummies(data=read_group_data, prefix='read_group', columns=['read_group'], drop_first=False)

subsample_sam['read_group'] = read_group_data['read_group_3'] # first column only - hardcoded because of dummy variable
# trap

# Split data into X and y to predict quality scores from other features

### Dependent variable - start with a single position being predicted by the metrics above

subsample_sam = subsample_sam.fillna(0)
subsample_sam = subsample_sam.sample(frac=0.01, axis=0, random_state=42)

subsample_sam = subsample_sam.loc[:, '210':'read_group']

y = subsample_sam['230']

### Dropping Y columns - need to figure out what columns to use!

X = subsample_sam.drop(['230'], axis=1)

# X = subsample_sam['read_group'] # temporary test

### May have to vary testing size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Observe shape of data

y_train = y_train.to_numpy()
y_train = y_train.reshape(len(y_train), 1)

y_test = y_test.to_numpy()
y_test = y_test.reshape(len(y_test), 1)

print('Shape of original dataset:      ', subsample_sam.shape)
print('Shape of input - training set:  ', X_train.shape)
print('Shape of output - training set: ', y_train.shape)
print('Shape of input - testing set:   ', X_test.shape)
print('Shape of output - testing set:  ', y_test.shape)

# Cross-validation  to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

# Define and fit ridge regressor

def normal_equation_ridge_regression(X, y, alpha):
    intercept_ones = np.ones((len(X), 1))        # results in array( [ [1],..,[1] ] )
    X_b = np.c_[intercept_ones, X]               # we now add the additional ones as a new column to our X
    I = np.identity(X_b.shape[1])               # identity matrix with dimensions (n+1)
    I[0][0] = 0                                 # adjusting the first value in I to be 0, to account for the intercept term
    theta_optimal = np.linalg.inv(X_b.T.dot(X_b) + alpha * I).dot(X_b.T).dot(y) # the normal equation
    return theta_optimal

ridge_model = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
ridge_model = ridge_model.fit(X_train, y_train)
theta_ridge_ne = np.array([ridge_model.intercept_, ridge_model.coef_[0]])

print('Alpha parameter:', ridge_model.alpha_)

def mse(y, y_predicted):
    error = y-y_predicted
    loss = 1/(y.size) * np.dot(error.T, error)
    return loss

def ridgeMSE(y, y_predicted, alpha, theta):
    mse = mse(y, y_predicted)
    ridge_mse = mse(y, y_predicted) + alpha * np.dot(theta,theta)
    return ridge_mse

def getRidgeMSEFunction(alpha=0.0001):

    def ridgeMSE(y, y_predicted, theta):
        mse_loss = mse(y, y_predicted)
        ridge_loss = mse_loss + alpha * np.dot(theta,theta)
        return ridge_loss

    return ridgeMSE

# Testing for overfitting

print('MSE for training data:', mse(y_train, ridge_model.predict(X_train)))
print('MSE for testing data:', mse(y_test, ridge_model.predict(X_test)))

# Testing for multicollinearity in X (for ridge regression)

VIF_results = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print('VIF for input data:', VIF_results)

# Plots

sns.set_theme(style='ticks')
heatmap_colormap = 'viridis'

qual_correlations = subsample_sam.corr()
plt.title('Correlation Matrix - Evidence for Multicollinearity', fontweight='bold')
sns.heatmap(qual_correlations, cmap=heatmap_colormap, center=0, square=True)
plt.show()

x_figs, (ax1, ax2) = plt.subplots(1, 2)

x1 = sns.heatmap(X_train, ax=ax1, cmap=heatmap_colormap, cbar=True, cbar_kws=dict(use_gridspec=False, location='left'))
x2 = sns.heatmap(X_test, ax=ax2, cmap=heatmap_colormap, cbar=False)
x1.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='positions along read (X train features)', ylabel='reads')
x2.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='positions along read (X test features)')

plt.show()

y_figs, (ax3, ax4, ax5) = plt.subplots(1, 3)

y1 = sns.heatmap(y_train, ax=ax3, cmap=heatmap_colormap, cbar=True, cbar_kws=dict(use_gridspec=False, location='left'))
y2 = sns.heatmap(y_test, ax=ax4, cmap=heatmap_colormap, cbar=False)
yp = sns.heatmap(data=ridge_model.predict(X_test), ax=ax5, cmap=heatmap_colormap, cbar=False) # y_pred
y1.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read (y train quality scores)', ylabel='reads')
y2.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read (y test quality scores)')
yp.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read (y pred quality scores)')

plt.show()

residuals = ridge_model.predict(X_test) - y_test
residuals = pd.DataFrame(residuals)
residuals = sns.heatmap(residuals.sample(frac=0.25), annot=True, cmap=heatmap_colormap) # residual matrix
residuals.set()

plt.show()
