import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

# Convert the ridge regressor data frame to a fastq file to run as a fastqc report
ridge_df = pd.read_csv('/Users/keshavgandhi/Downloads/ridge_regressor_data.csv')

### Drop index column (go back and add index=False to the ridge regressor)

ridge_df = ridge_df.drop(ridge_df.columns[0], axis=1)
output_fastq = 'ridge_output.fastq'

with open(output_fastq, 'w') as fq:

    for index, row in ridge_df.iterrows():

        values = row.values
        if pd.isna(values).any():
            values = [0] * len(values) # replace empty data

        seq = len(list(row.values.astype(int))) * 'A' # need specific length of sequence

        seq_record = SeqIO.SeqRecord(Seq(seq), id=str(index), description='')

        seq_record.letter_annotations['phred_quality'] = list(row.values.astype(int))
        SeqIO.write(seq_record, fq, 'fastq')

        # make a list of these four things
        # line1: name
        # line2: read
        # line3: "+"
        # line4: quality scores
