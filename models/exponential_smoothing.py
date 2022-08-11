# 8/10/22

import io
import os
import pysam
import math
import statistics
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

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

    # Append to master lists
    qual_list.append(align_qual)
    pos_list.append(ref_pos)
    i += 1
    j = print_update(i, modulo, j)

print(f'100% complete')
file_to_parse.close()

# Turn list of lists into a dataframe

subsample_sam = pd.DataFrame(qual_list)

# Pre-processing

column_names = list(np.arange(0, len(subsample_sam.columns), 1))
column_names = [dt.datetime.fromtimestamp(i) for i in column_names]
subsample_sam = subsample_sam.set_axis(column_names, axis='columns', inplace=False)

# Split data into X and y to predict quality scores from other features

### Dependent variable - start with a single position being predicted by the metrics above

subsample_sam = subsample_sam.dropna(axis=0)
sub_subsample_sam = subsample_sam.sample(frac=0.15, axis=0, random_state=42)

means_for_smoothing = sub_subsample_sam.mean()

double_exp_model = ExponentialSmoothing(means_for_smoothing, trend='additive', damped_trend=True, seasonal=None).fit()

yhat = double_exp_model.predict(0, len(sub_subsample_sam.columns) - 1)

predictions = pd.DataFrame(yhat).set_index(yhat.index)
predictions.columns = ['quality_scores']

plt.plot(predictions['quality_scores'])
plt.plot(means_for_smoothing)
plt.show()

heatmaps, (ax1, ax2) = plt.subplots(1, 2)

column_names = list(np.arange(0, len(subsample_sam.columns), 1))
subsample_sam = subsample_sam.set_axis(column_names, axis='columns', inplace=False)

actual_data = sns.heatmap(subsample_sam, vmin=0, vmax=40, cmap='viridis', ax=ax1, cbar=True, cbar_kws=dict(use_gridspec=False, location='left'))
actual_data.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read', ylabel='reads')

predictions = predictions.transpose()

column_names = list(np.arange(0, len(predictions.columns), 1))
predictions = predictions.set_axis(column_names, axis='columns', inplace=False)

repeats = len(qual_list)
predictions_repeat = pd.concat([predictions]*repeats, ignore_index=True)

predicted_data = sns.heatmap(predictions_repeat, vmin=0, vmax=40, cmap='viridis', ax=ax2, cbar=False)
predicted_data.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read', ylabel='reads')

plt.show()

residuals = np.subtract(subsample_sam, predictions_repeat)
sns.heatmap(residuals, vmin=0, vmax=40, cmap='viridis')
predicted_data.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='position along read', ylabel='reads')
plt.show()

mse_scores = np.square(residuals).mean()
rmse_scores = [math.sqrt(i) for i in mse_scores]
norm_rmse_scores = [i / 40 for i in rmse_scores]

sns.lineplot(data=mse_scores)
sns.lineplot(data=rmse_scores)
plt.show()

sns.lineplot(data=norm_rmse_scores)
plt.show()
