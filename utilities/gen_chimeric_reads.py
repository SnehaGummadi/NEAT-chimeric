# import pandas as pd

# TODO Get reads and place into an array.
# what if i write the chim file to a csv instead of a fastq file
# that'll give me more choices right????

def input_pre_chim_reads(chim_file_read1, chim_file_read2):
    i = 0

    temp_name = None
    read1 = None
    qual1 = None
    read2 = None
    qual2 = None

    with open(gunzip(chim_file_read1)) as open_read1 and open(gunzip(chim_file_read2)) as open_read2:
        for (line1 in open_read1) and (line2 in open_read2):
            if i % 4 == 0:
                temp_name = line1
            elif i % 4== 1:
                read1 = line1
                read2 = line2
            elif i % 4 == 3:
                qual1 = line1
                qual2 = line2
                output_file_writer.write_chim_csv(temp_name, read1, qual1, read2, qual2)





# TODO make chimeric when SV in read 1



# TODO make chimeric when SV in read 2



# Main class
class GenChimericReads:
    def __init__(self, chim_fq1, chim_fq2,):
        self.chim_fq1
    
