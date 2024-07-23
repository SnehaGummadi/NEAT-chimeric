import pandas as pd
import random
import logging

_LOG = logging.getLogger(__name__)

# TODO decide whether to select randomly or not.
    # randomly select the read to be made chimeric with a chance of 75%
    #rannum = random.randint(1,100)
    #if rannum <= 75:

# TODO Should alternate read be randomly selected?
    # If so, can add an extra column to determine if it has been used

# quality assigned during finalize_read_and_write

# Main class
class GenChimericReads:
    def __init__(self, in_read1, in_read2, in_read12, left, right, tename, count):
        header = ['read1', 'read2', ]
        self.made_chimeric_reads = pd.DataFrame(columns=header)
        self.chim_read_count = count
        self.tename = tename
        self.in_read1 = in_read1
        self.in_read2 = in_read2
        self.in_read12 = in_read12
        self.left = left
        self.right = right

        # So that reads new read pairs are not made with previously used reads
        self.in_read12_index_left_last = 0
        self.in_read12_index_right_last = 0


    def make_chim_reads(self):
        # Check that there are enough read pairs in left and right
        # TODO Account for dealing with too few values in left, right, and in_read12
        if len(self.left) < len(self.in_read12):
            _LOG.debug(f'Good on left')
        else:
            _LOG.debug(f'NOTTTTTTTTTT GOOOOOODDDDDDDDDD LEFTTTTTTTTTTTTTT')
        if len(self.right) < len(self.in_read12):
            _LOG.debug(f'good on right')
        else:
            _LOG.debug(f'NOTTTTTTTTTT GOOOOOODDDDDDDDDD RIGHTTTTTTTTTTTT')

        self.make_read1_chimeric()
        self.make_read2_chimeric()

        _LOG.info(f'Done making chimeric reads')


    def make_read1_chimeric(self):
        for index,row in self.left.iterrows():
            # Create new read name
            new_read_name = f'NEAT-generated_chr18_chim-read1-{self.tename}_{self.chim_read_count}'
        
            # Copy read 2 in left DF
            new_read2 = row['read2']

            # Get a read 1 that has TE in it
            new_read1 = self.in_read12.loc[self.in_read12_index_left_last]['read1']
            self.in_read12_index_left_last += 1

            # Change read names
            new_read1.name = f'{new_read_name}/1'
            new_read2.name = f'{new_read_name}/2'

            # Tell read that it is chimeric (so errors and mutations are not added to these reads)
            new_read1.is_chimeric = True
            new_read2.is_chimeric = True

            # Add row to DataFrame for the newly made chimeric reads
            new_row = [new_read1,new_read2]
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1

    def make_read2_chimeric(self):
        for index,row in self.right.iterrows():
            # Create new read name
            new_read_name = f'NEAT-generated_chr18_chim-read2-{self.tename}_{self.chim_read_count}'

            # Copy read 1 from right DF
            new_read1 = row['read1']

            # Get a read 2 that has TE in it
            new_read2 = self.in_read12.loc[self.in_read12_index_right_last]['read2']
            self.in_read12_index_right_last += 1

            # Change read names
            new_read1.name = f'{new_read_name}/1'
            new_read2.name = f'{new_read_name}/2'

            # Tell read that it is chimeric (so errors and mutations are not added to these reads)
            new_read1.is_chimeric = True
            new_read2.is_chimeric = True

            # Add row to DataFrame for the newly made chimeric reads
            new_row = [new_read1,new_read2]
            self.made_chimeric_reads.loc[len(self.made_chimeric_reads)] = new_row
            self.chim_read_count += 1        