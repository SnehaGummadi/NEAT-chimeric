import pysam
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

def make_qual_score_list(bam_file):

    '''Takes an input BAM file and creates lists of quality scores. This becomes a data frame, which will be
    pre-processed for ridge regression analysis.'''

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

    subsample_sam = pd.DataFrame(qual_list)

    # Pre-processing

    column_names = list(np.arange(0, len(subsample_sam.columns), 1))
    column_names = [str(i) for i in column_names]
    subsample_sam = subsample_sam.set_axis(column_names, axis='columns', inplace=False)

    # Adding features

    subsample_sam['unmap_mate'] = unmap_mate
    subsample_sam['sum'] = subsample_sam.sum(axis=1)
    subsample_sam['average'] = subsample_sam['sum'] / len(subsample_sam.columns)
    subsample_sam = subsample_sam.drop(columns=['sum'])

    # Adding read group as a feature - physicality of reads on slide can be used as a predictor

    file_to_parse = pysam.AlignmentFile(bam_file, check_sq=False)

    output_header = file_to_parse.header
    read_groups = output_header['RG']

    ### Each dictionary is a separate read group

    group_categories = {}

    for x, n in zip(read_groups, range(len(read_groups))):
        group_categories.update({x['ID']: n + 1})

    categories_by_record = []

    for entry in file_to_parse:
        group = entry.get_tag('RG')
        current_category = group_categories[group]
        categories_by_record.append(current_category)

    read_group_data = pd.DataFrame(categories_by_record, columns=['read_group'])
    read_group_data = pd.get_dummies(data=read_group_data, prefix='read_group', columns=['read_group'], drop_first=False)

    # Pre-processing - fill in missing data and sample 1% of reads

    subsample_sam = subsample_sam.fillna(0)
    subsample_sam = subsample_sam.sample(frac=0.01, axis=0, random_state=42)
    subsample_sam = subsample_sam.loc[:, '0':f'{len(subsample_sam.columns) - 3}'] # excludes features except position
    # subsample_sam = subsample_sam.loc[:, '0':'10']

    return subsample_sam

def ridge_regressor(data_frame):

    '''Goal of using a ridge regression model to iteratively predict quality scores for reads at every position in the
    genome. Predictor variables are all other quality scores, unmapped reads, and read group. Cross validation is
    optional and commented out.'''

    y_preds_list = []
    test_mse_list = []
    train_mse_list = []
    test_r2_list = []
    train_r2_list = []
    cv_accuracy_list = []

    for i in data_frame.columns:

        y = data_frame[i]
        X = data_frame.drop([i], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Cross-validation for model

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

        # Define and fit ridge regressor

        # ridge_model = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
        ridge_model = RidgeCV(alphas=[0, 1, 0.1], cv=cv, scoring='neg_mean_absolute_error')
        ridge_model = ridge_model.fit(X_train, y_train)

        test_r2 = ridge_model.score(X_test, y_test)
        train_r2 = ridge_model.score(X_train, y_train)

        test_r2_list.append(test_r2)
        train_r2_list.append(train_r2)

        accuracy_matrix = cross_val_score(ridge_model, X, y, cv=cv)
        accuracy = accuracy_matrix.mean() * 100
        cv_accuracy_list.append(accuracy)

        theta_ridge = ridge_model.intercept_, ridge_model.coef_[0]
        alpha_ridge = ridge_model.alpha_

        # Predicted columns

        y_pred = list(ridge_model.predict(X_train))
        y_preds_list.append(y_pred)

        mse_score_test = mean_squared_error(y_true=y_test, y_pred=ridge_model.predict(X_test))
        mse_score_train = mean_squared_error(y_true=y_train, y_pred=ridge_model.predict(X_train))

        test_mse_list.append(mse_score_test)
        train_mse_list.append(mse_score_train)

    return y_preds_list, test_mse_list, train_mse_list, test_r2_list, train_r2_list, cv_accuracy_list

def save_metrics(test_mse_list, train_mse_list, test_r2_list, train_r2_list, cv_accuracy_list):

    '''Prints metrics that measure cross validation and model performance. Will save to a file in a future update.'''

    print('mse test list:  ', test_mse_list)
    print('mse train list: ', train_mse_list)
    print('mse test mean:  ', np.mean(test_mse_list))
    print('mse train mean: ', np.mean(train_mse_list))

    print('r2 test list:   ', r2_score_test)
    print('r2 train list;  ', r2_score_train)
    print('r2 test mean:   ', np.mean(r2_score_test))
    print('r2 train mean:  ', np.mean(r2_score_train))

    print('accuracy:', cv_accuracies)

def plot_heatmap(y_preds_list, file_path):

    '''Takes a list of predicted y-values (for every position along the read) and plots a seaborn heatmap to visualize
    them.'''

    y_preds_list_df = pd.DataFrame(y_preds_list).transpose()

    sns.heatmap(y_preds_list_df, vmin=0, vmax=40, cmap='viridis')
    plt.savefig(f'{file_path}')

bam_file = '/Users/keshavgandhi/Downloads/subsample_3.125.bam'

test_df = make_qual_score_list(bam_file)
y_preds_test, mse_score_test, mse_score_train, r2_score_test, r2_score_train, cv_accuracies = ridge_regressor(test_df)

plot_heatmap(y_preds_test, '/Users/keshavgandhi/Downloads/test.svg')

# if __name__ == '__main__':
#     main()
