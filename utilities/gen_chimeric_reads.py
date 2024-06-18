import pandas as pd
import random

# TODO making chimeric reads when there are mulitple SV inserctions

# TODO decide whether to select randomly or not.
    # randomly select the read to be made chimeric with a chance of 75%
    #rannum = random.randint(1,100)
    #if rannum <= 75:

# TODO Should alternate read be randomly selected?
    # If so, can add an extra column to determine if it has been used

# TODO using output file writer, write reads to fq files

# Main class
class GenChimericReads:
    def __init__(self):
        header = ['read_name', 'read1', 'qual1', 'read2', 'qual2']
        self.made_chimeric_reads = pd.DataFrame(columns=header)
        self.chim_read_count = 1

        # So that reads new read pairs are not made with previously used reads
        self.left_index_left_off = 0
        self.right_index_left_off = 0


    def make_chim_reads(self, df_in_read1, df_in_read2, df_left_read1, df_in_read1_2, df_right_read2):
        self.make_read1_chimeric(df_in_read1, df_left_read1, df_in_read1_2)
        self.make_read2_chimeric(df_in_read2, df_right_read2, df_in_read1_2)


    def make_read1_chimeric(self, df_in_read1, df_left_read1, df_in_read1_2):

        # TODO Account for multiple different insertions in the DataFrames
        # interate through the length of the reads in df_in_read1
        for i in range(len(df_in_read1)):
            #Change read name 
            # TODO find a way to include the beginning read name info
            new_read_name = f'@chim_read-{self.chim_read_count}-read1SV'

            # Copy read 1 (which has the SV)
            new_read1 = df_in_read1.at[i,'read1']
            new_qual1 = df_in_read1.at[i,'qual1']

            # Get a read that has SV left of it's read 1
            
            new_read2 = df_left_read1.at[self.left_index_left_off,'read2']
            new_qual2 = df_left_read1.at[self.left_index_left_off,'qual2']
            self.left_index_left_off += 1

            # Add row to DataFrame for the newly made chimeric reads
            new_row = {'read_name': new_read_name, 'read1': new_read1, 'qual1': new_qual1, 'read2': new_read2, 'qual2': new_qual2}
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1
        
        for i in range(len(df_in_read1_2)):
            # Create a new read name
            new_read_name = f'@chim_read-{self.chim_read_count}-read1SV'

            # Copy read 1 (which has the SV)
            new_read1 = df_in_read1_2.at[i, 'read1']
            new_qual1 = df_in_read1_2.at[i, 'qual1']

            # Get a read that has the SV to the left of it's read 1
            new_read2 = df_left_read1.at[self.left_index_left_off,'read2']
            new_qual2 = df_left_read1.at[self.left_index_left_off,'qual2']
            self.left_index_left_off += 1

            # Add row to DataFrame for the newly made chimeric reads
            new_read_name = '@chim_read-' + self.chim_read_count + '-read1SV'
            new_row = {'read_name': new_read_name, 'read1': new_read1, 'qual1': new_qual1, 'read2': new_read2, 'qual2': new_qual2}
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1


    def make_read2_chimeric(self, df_in_read2, df_right_read2, df_in_read1_2):
        #iterate through the length of the reads in df_in_read2
        for i in range(len(df_in_read2)):
            # Create a new read name
            new_read_name = '@chim_read-' + self.chim_read_count + '-read2SV'

            # Copy read 2 (which has the SV)
            new_read2 = df_in_read2.at[i,'read2']
            new_qual2 = df_in_read2.at[i,'qual2']

            # Get a read that has the SV to the right of its read 2
            new_read1 = df_right_read2.at[self.right_index_left_off,'read1']
            new_qual1 = df_right_read2.at[self.right_index_left_off,'qual1']
            self.right_index_left_off += 1

            # Add row to DataFrame for the newly made chimeric reads
            new_row = {'read_name': new_read_name, 'read1': new_read1, 'qual1': new_qual1, 'read2': new_read2, 'qual2': new_qual2}
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1

        for i in range(len(df_in_read1_2)):
            # Create a new read name
            new_read_name = '@chim_read-' + self.chim_read_count + '-read2SV'

            # Copy read 2 (which has the SV)
            new_read2 = df_in_read1_2.at[i,'read2']
            new_qual2 = df_in_read1_2.at[i,'qual2']

            # Get a read that has the SV to the right of its read 2
            new_read1 = df_right_read2.at[self.right_index_left_off,'read1']
            new_qual1 = df_right_read2.at[self.right_index_left_off,'qual1']
            self.right_index_left_off += 1

            # Add row to DataFrame for the newly made chimeric reads
            new_row = {'read_name': new_read_name, 'read1': new_read1, 'qual1': new_qual1, 'read2': new_read2, 'qual2': new_qual2}
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1


    def write_to_fq(self):
        # Write chimeric reads to fq
        self.made_chimeric_reads

        # Write the remaining reads in df_left_read1 and df_right_read2 to fq