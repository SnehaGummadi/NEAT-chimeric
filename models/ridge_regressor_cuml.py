import pysam
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import pickle

import cudf
import cuml
import cupy as cp
from cuml.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from cuml.metrics.regression import mean_squared_error
from cuml.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

def make_qual_score_list(bam_file):

    '''Takes an input BAM file and creates lists of quality scores. This becomes a data frame, which will be
    pre-processed for ridge regression analysis.'''

    index = f'{bam_file}.bai'

    if not pathlib.Path(index).exists():
        print('No index found, creating one.')
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

    print('Parsing file')

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

        # Mapping reference positions
        ref_pos = item.get_reference_positions()

        # Mapping lack of paired reads (to binarize)
        unmap_mate = int(item.mate_is_unmapped)

        # Append to master lists
        qual_list.append(align_qual)
        unmap_list.append(unmap_mate)
        pos_list.append(ref_pos)
        i += 1
        j = print_update(i, modulo, j)

    print(f'100% complete')
    file_to_parse.close()

    # Turn list of lists into a dataframe

    quality_df = cudf.DataFrame(qual_list)

    # Adding features

    quality_df['unmap_mate'] = pd.Series(unmap_list)
    quality_df['sum'] = quality_df.sum(axis=1)
    quality_df['average'] = quality_df['sum'] / len(quality_df.columns)
    quality_df = quality_df.drop(columns=['sum'])

    # Adding read group as a feature - physicality of reads on slide can be used as a predictor

    file_to_parse = pysam.AlignmentFile(bam_file, check_sq=False)

    output_header = file_to_parse.header
    read_groups = output_header['RG']

    # Each dictionary is a separate read group

    group_categories = {}

    for x, n in zip(read_groups, range(len(read_groups))):
        group_categories.update({x['ID']: n + 1})

    categories_by_record = []

    for entry in file_to_parse:
        group = entry.get_tag('RG')
        current_category = group_categories[group]
        categories_by_record.append(current_category)

    read_group_data = cudf.DataFrame(categories_by_record, columns=['read_group'])
    read_group_data = cudf.get_dummies(read_group_data, prefix='read_group', columns=['read_group'], drop_first=False)

    # Pre-processing - fill in missing data and sample 1% of reads

    quality_df = quality_df.fillna(0)
    quality_df = quality_df.sample(frac=0.01, axis=0, random_state=42)
    quality_df = quality_df.iloc[:,:-3] # excludes features except position (for testing purposes)

    return quality_df

def make_df(data_frame, column):

    y = data_frame[column]
    X = data_frame.drop([column], axis=1)

    return X, y

def fit_ridge_regressor(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Cross-validation for model

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    # Define and fit ridge regressor

    ridge_model = RidgeCV(alphas=np.arange(0.1, 1.0, 0.1), cv=cv, scoring='neg_mean_absolute_error')

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    
    ridge_model = ridge_model.fit(X_train, y_train)

    return X, y, ridge_model, X_train, X_test, y_train, y_test

def fit_and_save_ridge_regressor(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Cross-validation for model

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    # Define and fit ridge regressor

    ridge_model = RidgeCV(alphas=np.arange(0.1, 1.0, 0.1), cv=cv, scoring='neg_mean_absolute_error')

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    ridge_model = ridge_model.fit(X_train, y_train)

    # Save ridge_model to a pickle file

    with open('ridge_model.pickle', 'wb') as f:
        pickle.dump(ridge_model, f)
    
    print('Ridge regressor saved')

def fit_and_save_ridge_regressor_list(data_frame):

    ridge_model_list = []
    
    for i in data_frame.columns:
    
        X, y = make_df(data_frame, i)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Cross-validation for model

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

        # Define and fit ridge regressor

        ridge_model = RidgeCV(alphas=np.arange(0.1, 1.0, 0.1), cv=cv, scoring='neg_mean_absolute_error')

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        ridge_model = ridge_model.fit(X_train, y_train)
        ridge_model_list.append(ridge_model)

    # Save ridge_model to a pickle file

    with open('ridge_model_list.pickle', 'wb') as f:
        pickle.dump(ridge_model_list, f)

    print('Ridge regressor list saved')

def load_ridge_model_from_pickle(file_path):
        
    # Open the pickle file in read binary mode

    with open(file_path, 'rb') as f:
        ridge_model = pickle.load(f)
    
    return ridge_model

def load_ridge_model_list_from_pickle(file_path):

    # Open the pickle file in read binary mode

    with open(file_path, 'rb') as f:
        ridge_model_list = pickle.load(f)

    print('Successfully loaded a list of ridge regressors')

    return ridge_model_list

def metrics_ridge_regressor(X, y, ridge_model, X_train, X_test, y_train, y_test):

    test_r2 = ridge_model.score(X_test, y_test)
    train_r2 = ridge_model.score(X_train, y_train)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    theta_ridge = ridge_model.intercept_, ridge_model.coef_[0]
    alpha_ridge = ridge_model.alpha_

    return test_r2, train_r2, theta_ridge, alpha_ridge

def pred_ridge_regressor(ridge_model, X_train, X_test, y_train, y_test):

    y_pred = list(ridge_model.predict(X_train))

    mse_score_test = mean_squared_error(y_true=y_test, y_pred=ridge_model.predict(X_test))
    mse_score_train = mean_squared_error(y_true=y_train, y_pred=ridge_model.predict(X_train))

    return y_pred, mse_score_test, mse_score_train

def test_ridge_models(X_test, y_test, ridge_model_list):

    # Load the list of ridge models from the pickle file

    model_list = load_ridge_model_list_from_pickle('ridge_model_list.pickle')
    y_preds_list = []

    for i, model in enumerate(model_list):

        y_pred = model.predict(X_test)
        y_preds_list.append(y_pred)

    print('Ridge model list successfully used')

    return y_preds_list

def save_metrics(test_mse_list, train_mse_list, test_r2_list, train_r2_list, file_path):

    '''Prints metrics that measure cross validation and model performance, and saves them to a file.'''

    with open(file_path, 'w') as f:

        f.write('mse test list:  {}\n'.format(test_mse_list))
        f.write('mse train list: {}\n'.format(train_mse_list))
        f.write('mse test mean:  {}\n'.format(np.mean(test_mse_list)))
        f.write('mse train mean: {}\n'.format(np.mean(train_mse_list)))

        f.write('r2 test list:   {}\n'.format(test_r2_list))
        f.write('r2 train list:  {}\n'.format(train_r2_list))
        f.write('r2 test mean:   {}\n'.format(np.mean(test_r2_list)))
        f.write('r2 train mean:  {}\n'.format(np.mean(train_r2_list)))

    print('Metrics saved to file: {}'.format(file_path))

def plot_heatmap(y_preds_list, file_path):

    '''Takes a list of predicted y-values (for every position along the read) and plots a seaborn heatmap to visualize
    them.'''

    y_preds_list_df = pd.DataFrame(y_preds_list).transpose()

    with open('/projects/neat/krg3/ridge_regressor_results_more_fxns.txt', 'w') as csv_file:
        y_preds_list_df.to_csv(path_or_buf=csv_file) # temporary

    sns.heatmap(y_preds_list_df, vmin=0, vmax=40, cmap='viridis')
    plt.savefig(f'{file_path}')

    print('Heatmap plotted')

# bam_file = '/projects/neat/krg3/subsample.3.125.bam' # 18 seconds for 3.125, 24 seconds for 6.25, 32 seconds for 12.5, 55 seconds for 25, 89 seconds for 50, 189 seconds for full
bam_file = '/projects/neat/krg3/normal_sample.6.25.bam' # 113 seconds for 3.125, 
test_df = make_qual_score_list(bam_file)
# y_preds_test, mse_score_test, mse_score_train = use_ridge_regressor(test_df)

X, y = make_df(test_df, 42)
X, y, ridge_model, X_train, X_test, y_train, y_test = fit_ridge_regressor(X, y)
# fit_and_save_ridge_regressor_list(test_df)
ridge_model_list = load_ridge_model_list_from_pickle('ridge_model_list.pickle')
y_preds_list = test_ridge_models(X_test, y_test, ridge_model_list)
plot_heatmap(y_preds_list, 'test_ridge_models_fxn_test.svg')
