# Open a FASTA file for reading
with open("/home/sag200006/scratch/NEAT-chimeric/reference_files/Homo_sapiens.GRCh38.dna.chromosome.18.fa", "r") as input_fa:  
    # Open a FASTA file for writing
    with open("/home/sag200006/scratch/NEAT-chimeric/reference_files/chr18_smallest.fa", "w") as output_file:
        # Write sequence data to the output file
        # Iterate through each line in the input FASTA file
        identifier = None
        i=0
        for line in input_fa:
            if i < 200000:
                output_file.write(line)
            else:
                break
            i+=1

