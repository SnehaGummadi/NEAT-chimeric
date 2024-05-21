import pandas as pd

# TODO Get reads and place into an array.
# what if i write the chim file to a csv instead of a fastq file
# that'll give me more choices right????

def input_pre_chim_reads(chim_csv):
    chim_reads_df = pd.read_csv(chim_csv)
    
    i=0
    for read_name in chim_reads_df[0][i]:
        if read_name.contains('in-read1'):
            pd.DataFrame(chim_reads_df.row(i))
        if read_name.contains('in-read2'):
            chim_reads_df
        if read_name.contains('-in-read-1-and-2'):
            chim_reads_df
        i += 1

            


# TODO make chimeric when SV in read 1



# TODO make chimeric when SV in read 2



# Main class
class GenChimericReads:
    def __init__(self, chim_fq1, chim_fq2,):
        self.chim_fq1
    
