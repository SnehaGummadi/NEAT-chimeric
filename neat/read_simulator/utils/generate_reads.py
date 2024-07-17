import logging
import time
import pickle
import sys
import pandas as pd
import random

from math import ceil
from pathlib import Path
from Bio import SeqRecord
from bisect import bisect_left, bisect_right

from ...models import SequencingErrorModel, FragmentLengthModel, MutationModel
from .options import Options
from ...common import open_output
from ...variants import ContigVariants
from .read import Read
from .gen_chim_reads import GenChimericReads

__all__ = [
    'generate_reads',
    'cover_dataset',
    'overlaps',
]

_LOG = logging.getLogger(__name__)


def cover_dataset(
        span_length: int,
        options: Options,
        fragment_model: FragmentLengthModel | None,
) -> list:
    """
    Covers a dataset to the desired depth in the paired ended case. This is the main algorithm for creating the reads
    to the proper coverage depth. It uses an abstract representation of the reads, by end points.

    :param span_length: The total length the cover needs to span
    :param options: The options for the run
    :param fragment_model: The fragment model used for to generate random fragment lengths
    """

    final_reads = set()
    # sanity check
    if span_length/fragment_model.fragment_mean < 5:
        _LOG.warning("The fragment mean is relatively large compared to the chromosome size. You may need to increase "
                     "standard deviation, or decrease fragment mean, if NEAT cannot complete successfully.")
    # precompute how many reads we want
    # The numerator is the total number of base pair calls needed.
    # Divide that by read length gives the number of reads needed
    number_reads = ceil((span_length * options.coverage) / options.read_len)

    # We use fragments to model the DNA
    fragment_pool = fragment_model.generate_fragments(number_reads * 3)

    # step 1: Divide the span up into segments drawn from the fragment pool. Assign reads based on that.
    # step 2: repeat above until number of reads exceeds number_reads * 1.5
    # step 3: shuffle pool, then draw number_reads (or number_reads/2 for paired ended) reads to be our reads
    read_count = 0
    loop_count = 0
    while read_count <= number_reads:
        start = 0
        loop_count += 1
        # if loop_count > options.coverage * 100:
        #     _LOG.error("The selected fragment mean and standard deviation are causing NEAT to get stuck.")
        #     _LOG.error("Please try adjusting fragment mean or standard deviation to see if that fixes the issue.")
        #     _LOG.error(f"parameters:\n"
        #                f"chromosome length: {span_length}\n"
        #                f"read length: {options.read_len}\n"
        #                f"fragment mean: {options.fragment_mean}\n"
        #                f"fragment standard deviation: {options.fragment_st_dev}")
        #     sys.exit(1)
        temp_fragments = []
        # trying to get enough variability to harden NEAT against edge cases.
        if loop_count % 10 == 0:
            fragment_model.rng.shuffle(fragment_pool)
        # Breaking the gename into fragments
        while start < span_length:
            # We take the first element and put it back on the end to create an endless pool of fragments to draw from
            fragment = fragment_pool.pop(0)
            end = min(start + fragment, span_length)
            # these are equivalent of reads we expect the machine to filter out, but we won't actually use it
            if end - start < options.read_len:
                # add some random flavor to try to keep it to falling into a loop
                if fragment_model.rng.normal() < 0.5:
                    fragment_pool.insert(len(fragment_pool)//2, fragment)
                else:
                    fragment_pool.insert(len(fragment_pool) - 3, fragment)
            else:
                fragment_pool.append(fragment)
                temp_fragments.append((start, end))
            start = end

        # Generating reads from fragments
        for fragment in temp_fragments:
            read_start = fragment[0]
            read_end = read_start + options.read_len
            # This filters out those small fragments, to give the dataset some realistic variety
            if read_end > fragment[1]:
                continue
            else:
                read1 = (read_start, read_end)
                if options.paired_ended:
                    # This will be valid because of the check above
                    read2 = (fragment[1] - options.read_len, fragment[1])
                else:
                    read2 = (0, 0)
                # The structure for these reads will be (left_start, left_end, right_start, right_end)
                # where start and end are ints with end > start. Reads can overlap, so right_start < left_end
                # is possible, but the reads cannot extend past each other, so right_start < left_start and
                # left_end > right_end are not possible.

                # sanity check that we haven't created an unrealistic read:
                insert_size = read2[0] - read1[1]
                if insert_size > 2 * options.read_len:
                    # Probably an outlier fragment length. We'll just pitch one of the reads
                    # and consider it lost to the ages.
                    if fragment_model.rng.choice((True, False)):
                        read1 = (0, 0)
                    else:
                        read2 = (0, 0)
                read = read1 + read2
                if read not in final_reads:
                    final_reads.add(read)
                    read_count += 1

    # Convert set to final list
    final_reads = list(final_reads)
    # Now we shuffle them to add some randomness
    fragment_model.rng.shuffle(final_reads)
    # And only return the number we needed
    _LOG.debug(f"Coverage required {loop_count} loops")
    if options.paired_ended:
        # Since each read is actually 2 reads, we only need to return half as many. But to cheat a few extra, we scale
        # that down slightly to 1.85 reads per read. This factor is arbitrary and may even be a function. But let's see
        # how well this estimate works
        return final_reads[:ceil(number_reads/1.85)]
    else:
        # Each read lacks a pair, so we need the full number of single ended reads
        return final_reads[:number_reads]


def find_applicable_mutations(my_read: Read, all_variants: ContigVariants) -> dict:
    """
    Scans the variants' dict for appropriate mutations.

    :param my_read: The read object to add the mutations to
    :param all_variants: All the variants for the dataset
    :return: A list of relevant mutations
    """
    return_dict = {}
    left = bisect_left(all_variants.variant_locations, my_read.position)
    right = bisect_right(all_variants.variant_locations, my_read.end_point - 1)
    subset = all_variants.variant_locations[left: right]
    for index in subset:
        return_dict[index] = all_variants[index]
    return return_dict


def overlaps(test_interval: tuple[int, int], comparison_interval: tuple[int, int]) -> bool:
    """
    This function checks if the read overlaps with an input interval.
    :param test_interval: the interval to test, expressed as a tuple of end points
        (understood to be a half-open interval)
    :param comparison_interval: the interval to check against, expressed as a tuple of end points

    Four situations where we can say there is an overlap:
       1. The comparison interval contains the test interval start point
       2. The comparison interval contains the test interval end point
       3. The comparison interval contains both start and end points of the test interval
       4. The comparison interval is within the test interval
    Although 3 is really just a special case of 1 and 2, so we don't need a separate check

    If the read is equal to the interval, then all of these will be trivially true,
    and we don't need a separate check.
    """
    return (comparison_interval[0] < test_interval[1] < comparison_interval[1]) or \
           (comparison_interval[0] <= test_interval[0] < comparison_interval[1]) or \
           (test_interval[0] <= comparison_interval[0] and test_interval[1] >= comparison_interval[1])


def generate_reads(reference: SeqRecord,
                   reads_pickle: str,
                   error_model_1: SequencingErrorModel,
                   error_model_2: SequencingErrorModel | None,
                   mutation_model: MutationModel,
                   fraglen_model: FragmentLengthModel,
                   contig_variants: ContigVariants,
                   temporary_directory: str | Path,
                   targeted_regions: list,
                   discarded_regions: list,
                   options: Options,
                   chrom: str,
                   ref_start: int = 0
                   ) -> tuple:
    """
    This will generate reads given a set of parameters for the run. The reads will output in a fastq.

    :param reference: The reference segment that reads will be drawn from.
    :param reads_pickle: The file to put the reads generated into, for bam creation.
    :param error_model_1: The error model for this run, the forward strand
    :param error_model_2: The error model for this run, reverse strand
    :param mutation_model: The mutation model for this run
    :param fraglen_model: The fragment length model for this run
    :param contig_variants: An object containing all input and randomly generated variants to be included.
    :param temporary_directory: The directory where to store temporary files for the run
    :param targeted_regions: A list of regions to target for the run (at a rate defined in the options
        file or 2% retained by default)
    :param discarded_regions: A list of regions to discard for the run
    :param options: The options entered for this run by the user
    :param chrom: The chromosome this reference segment originates from
    :param ref_start: The start point for this reference segment. Default is 0 and this is currently not fully
        implemented, to be used for parallelization.
    :return: A tuple of the filenames for the temp files created
    """
    # Set up files for use. May not need r2, but it's there if we do.
    # We will separate the properly paired and the singletons.
    # For now, we are making an assumption that the chromosome name contains no invalid characters for bash file names
    # such as `*` or `:` even though those are technically allowed.
    # TODO We'll need to add some checks to ensure that this is the case.
    chrom_fastq_r1_paired = temporary_directory / f'{chrom}_r1_paired.fq.bgz'
    chrom_fastq_r1_single = temporary_directory / f'{chrom}_r1_single.fq.bgz'
    chrom_fastq_r2_paired = temporary_directory / f'{chrom}_r2_paired.fq.bgz'
    chrom_fastq_r2_single = temporary_directory / f'{chrom}_r2_single.fq.bgz'

    _LOG.info(f'Sampling reads...')
    start_time = time.time()

    base_name = f'NEAT-generated_{chrom}'

    _LOG.debug("Covering dataset.")
    t = time.time()
    reads = cover_dataset(
        len(reference),
        options,
        fraglen_model,
    )
    _LOG.debug(f"Dataset coverage took: {(time.time() - t)/60:.2f} m")

    # These will hold the values as inserted.
    properly_paired_reads = []
    singletons = []

    """
                for line_hervk_svaa
                line at 2573000 - 2579053
                hervk at 3461053 - 3468589
                sva_a at 9152589 - 9153976

                for alu_y_y_sz_jb
                aluY at 811000 - 811311
                aluY at 2573311 - 2573622
                aluSz at 5294122 - 5294434
                aluJb at 9139934 - 9140246
                """
    column = ['name','start','end']
    lhs_data = [['line',2573000,2579053],['hervk',3461053,3468589],['svaa',9152589,9153976]]
    alu_data = [['y1',811000,811311],['y2',2573311,2573622],['sz',5294122,5294434],['jb',9139934,9140246]]
    lhs_ins = pd.DataFrame(lhs_data,columns=column)
    alu_ins = pd.DataFrame(alu_data,columns=column)

    read_column = ['read1','read2']
    left_line = pd.DataFrame(columns=read_column)
    right_line = pd.DataFrame(columns=read_column)
    line_read1 = pd.DataFrame(columns=read_column)
    line_read2 = pd.DataFrame(columns=read_column)
    line_read12 = pd.DataFrame(columns=read_column)

    left_hervk = pd.DataFrame(columns=read_column)
    right_hervk = pd.DataFrame(columns=read_column)
    hervk_read1 = pd.DataFrame(columns=read_column)
    hervk_read2 = pd.DataFrame(columns=read_column)
    hervk_read12 = pd.DataFrame(columns=read_column)

    left_svaa = pd.DataFrame(columns=read_column)
    right_svaa = pd.DataFrame(columns=read_column)
    svaa_read1 = pd.DataFrame(columns=read_column)
    svaa_read2 = pd.DataFrame(columns=read_column)
    svaa_read12 = pd.DataFrame(columns=read_column)

    print(f'reference_id: {reference.id}, chrom: {chrom}, ref_start: {ref_start}')

    _LOG.debug("Writing fastq(s) and optional tsam, if indicated")
    t = time.time()
    with (
        open_output(chrom_fastq_r1_paired) as fq1_paired,
        open_output(chrom_fastq_r1_single) as fq1_single,
        open_output(chrom_fastq_r2_paired) as fq2_paired,
        open_output(chrom_fastq_r2_single) as fq2_single
    ):

        for i in range(len(reads)):
            which_read = -1
            sent_to_chimeric = False
            span_1 = False
            span_2 = False
            print(f'{i/len(reads):.2%}', end='\r')
            # First thing we'll do is check to see if this read is filtered out by a bed file
            read1, read2 = (reads[i][0], reads[i][1]), (reads[i][2], reads[i][3])
            found_read1, found_read2 = False, False
            # For no target bed, there wil only be one region to check and this will complete very quickly
            for region in targeted_regions:
                # If this is a false region, we can skip it
                if not region[2]:
                    continue
                # We need to make sure this hasn't been filtered already, so if any coordinate is nonzero (falsey)
                if any(read1):
                    # Check if read1 is in this targeted region (any part of it overlaps)
                    if overlaps(read1, (region[0], region[1])):
                        found_read1 = True
                # Again, make sure it hasn't been filtered, or this is a single-ended read
                if any(read2):
                    if overlaps(read2, (region[0], region[1])):
                        found_read2 = True
            # This read was outside targeted regions
            if not found_read1:
                # Filter out this read
                read1 = (0, 0)
            if not found_read2:
                # Note that for single ended reads, it will never find read2 and this does nothing (it's already (0,0))
                read2 = (0, 0)

            # If there was no discard bed, this will complete very quickly
            discard_read1, discard_read2 = False, False
            for region in discarded_regions:
                # If region[2] is False then this region is not being discarded and we can skip it
                if not region[2]:
                    continue
                # Check to make sure the read isn't already filtered out
                if any(read1):
                    if overlaps(read1, (region[0], region[1])):
                        discard_read1 = True
                # No need to worry about already filtered reads
                if any(read2):
                    if overlaps(read2, (region[0], region[1])):
                        discard_read2 = True
            if discard_read1:
                read1 = (0, 0)
            if discard_read2:
                read2 = (0, 0)

            # This raw read will replace the original reads[i], and is the raw read with filters applied.
            raw_read = read1 + read2

            # If both reads were filtered out, we can move along
            if not any(raw_read):
                continue
            else:
                # We must have at least 1 read that has data
                # If only read1 or read2 is absent, this is a singleton
                read1_is_singleton = False
                read2_is_singleton = False
                properly_paired = False
                if not any(read2):
                    # Note that this includes all single ended reads that passed filter
                    read1_is_singleton = True
                elif not any(read1):
                    # read1 got filtered out
                    read2_is_singleton = True
                else:
                    properly_paired = True

            read_name = f'{base_name}_{str(i+1)}'

            # If the other read is marked as a singleton, then this one was filtered out, or these are single-ended
            if not read2_is_singleton:
                # It's properly paired if it's not a singleton
                is_paired = not read1_is_singleton
                # add a small amount of padding to the end to account for deletions.
                # Trying out this method of using the read-length, which for the default neat run gives ~30.
                padding = options.read_len//5
                segment = reference[read1[0]: read1[1] + padding].seq

                # if we're at the end of the contig, this may not pick up the full padding
                actual_padding = len(segment) - options.read_len

                read_1 = Read(
                    name=read_name + "/1",
                    raw_read=raw_read,
                    reference_segment=segment,
                    reference_id=reference.id,
                    position=read1[0] + ref_start,
                    end_point=read1[1] + ref_start,
                    padding=actual_padding,
                    is_paired=is_paired,
                    is_read1=True
                )

                read_1.mutations = find_applicable_mutations(read_1, contig_variants)
                if is_paired:
                    handle = fq1_paired
                else:
                    handle = fq1_single
                

            # if read1 is a sinleton then these are single-ended reads or this one was filtered out, se we skip
            if not read1_is_singleton:
                is_paired = not read2_is_singleton
                # Padding, as above
                padding = options.read_len//5
                start_coordinate = max((read2[0] - padding), 0)
                # this ensures that we get a segment with NEAT-recognized bases
                segment = reference[start_coordinate: read2[1]].seq
                # See note above
                actual_padding = len(segment) - options.read_len

                read_2 = Read(
                    name=read_name + "/2",
                    raw_read=reads[i],
                    reference_segment=segment,
                    reference_id=reference.id,
                    position=read2[0] + ref_start,
                    end_point=read2[1] + ref_start,
                    padding=actual_padding,
                    is_reverse=True,
                    is_paired=is_paired,
                    is_read1=False
                )

                read_2.mutations = find_applicable_mutations(read_2, contig_variants)
                if is_paired:
                    handle = fq2_paired
                else:
                    handle = fq2_single    

            if properly_paired:
                for index,row in lhs_ins.iterrows():

                    if(row['end']+200 < read_1.position and row['end']+1000 > read_1.position):
                        # TE is entirely to the left of read 1
                        #rannum = random.randint(1,100)
                        #if rannum < 25:
                            sent_to_chimeric = True
                            if 'line' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-line-left/1"
                                read_2.name = f"{og_name_2[:-2]}-line-left/2"
                                left_line.loc[len(left_line)] = [read_1,read_2]
                            elif 'hervk' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-hervk-left/1"
                                read_2.name = f"{og_name_2[:-2]}-hervk-left/2"
                                left_hervk.loc[len(left_hervk)] = [read_1,read_2]
                            elif 'svaa' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-svaa-left/1"
                                read_2.name = f"{og_name_2[:-2]}-svaa-left/2"
                                left_svaa.loc[len(left_svaa)] = [read_1,read_2]
            
                    if(max(read_1.position,row['start']) <= min(read_1.end_point,row['end'])):
                        # TE is either partially or entirely in read 1
                        which_read = 1
                        sent_to_chimeric = False
                        if(read_1.position >= row['start'] and read_1.end_point <= row['end']):
                            # TE spans read 1
                            span_1 = True
            
                    # if(read_1.end_point < row['start'] and row['end'] < read_2.position):
                        # TE is entirely between read 1 and read 2
                        # nothing to do
                    
                    if(max(read_2.position,row['start']) <= min(read_2.end_point,row['end'])):
                        # TE is entirely or paritially within read 2
                        which_read = 2
                        sent_to_chimeric = False
                        if(read_2.position >= row['start'] and read_2.end_point <= row['end']):
                            # TE spans read 2
                            span_2 = True
                    
                    if(row['start'] > read_2.end_point+200 and row['start'] < read_2.end_point+1000):
                        # TE is entirely to the right of read 2
                        #rannum = random.randint(1,100)
                        #if rannum < 75:
                            sent_to_chimeric = True
                            if 'line' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-line-right/1"
                                read_2.name = f"{og_name_2[:-2]}-line-right/2"
                                right_line.loc[len(right_line)] = [read_1,read_2]
                            elif 'hervk' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-hervk-right/1"
                                read_2.name = f"{og_name_2[:-2]}-hervk-right/2"
                                right_hervk.loc[len(right_hervk)] = [read_1,read_2]
                            elif 'svaa' in row['name']:
                                og_name_1 = read_1.name
                                og_name_2 = read_2.name
                                read_1.name = f"{og_name_1[:-2]}-svaa-right/1"
                                read_2.name = f"{og_name_2[:-2]}-svaa-right/2"
                                right_svaa.loc[len(right_svaa)] = [read_1,read_2]
                    
                    if(max(read_1.position,row['start']) <= min(read_1.end_point,row['end']) and
                       max(read_2.position,row['start']) <= min(read_2.end_point,row['end'])):
                        # TE is present entirely or partially in both read 1 and read 2
                        which_read = 3
                        sent_to_chimeric = True
                    
                    if which_read == 1:
                        if span_1 is True:
                            read_1.name = f"{read_1.name[:-2]}-span1/1"
                            read_2.name = f"{read_2.name[:-2]}-span1/2"
                        read_1.name = f"{read_1.name[:-2]}-in-read1-{row['name']}/1"
                        read_2.name = f"{read_2.name[:-2]}-in-read1-{row['name']}/2"
                        if 'line' in row['name']:
                            line_read1.loc[len(line_read1)] = [read_1,read_2]
                        elif 'hervk' in row['name']:
                            hervk_read1.loc[len(hervk_read1)] = [read_1,read_2]
                        elif 'svaa' in row['name']:
                            svaa_read1.loc[len(svaa_read1)] = [read_1,read_2]
                    elif which_read == 2:
                        if span_2 is True:
                            read_1.name = f"{read_1.name[:-2]}-span2/1"
                            read_2.name = f"{read_2.name[:-2]}-span2/2"
                        read_1.name = f"{read_1.name[:-2]}-in-read2-{row['name']}/1"
                        read_2.name = f"{read_2.name[:-2]}-in-read2-{row['name']}/2"
                        if 'line' in row['name']:
                            line_read2.loc[len(line_read2)] = [read_1,read_2]
                        elif 'hervk' in row['name']:
                            hervk_read2.loc[len(hervk_read2)] = [read_1,read_2]
                        elif 'svaa' in row['name']:
                            svaa_read2.loc[len(svaa_read2)] = [read_1,read_2]
                    elif which_read == 3:
                        if span_1 is True:
                            read_1.name = f"{read_1.name[:-2]}-span1/1"
                            read_2.name = f"{read_2.name[:-2]}-span1/2"
                        if span_2 is True:
                            read_1.name = f"{read_1.name[:-2]}-span2/1"
                            read_2.name = f"{read_2.name[:-2]}-span2/2"
                        read_1.name = f"{read_1.name[:-2]}-in-read-1-2-{row['name']}/1"
                        read_2.name = f"{read_2.name[:-2]}-in-read-1-2-{row['name']}/2"
                        if 'line' in row['name']:
                            line_read12.loc[len(line_read12)] = [read_1,read_2]
                        elif 'hervk' in row['name']:
                            hervk_read12.loc[len(hervk_read12)] = [read_1,read_2]
                        elif 'svaa' in row['name']:
                            svaa_read12.loc[len(svaa_read12)] = [read_1,read_2]
                    which_read = -1

                # Send the reads to temp fastqs
                # This is assuming that we require paired end reads otherwise will fail here
                if sent_to_chimeric == False:
                    read_1.finalize_read_and_write(
                        error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
                    )
                    read_2.finalize_read_and_write(
                        error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
                    )
                    properly_paired_reads.append((read_1, read_2))
            elif read1_is_singleton:
                # This will be the choice for all single-ended reads
                singletons.append((read_1, None))
            else:
                singletons.append((None, read_2))

        _LOG.debug("Generating chimeric reads")
        start = len(reads)
        # Create instance of the GenChimericReads class for each TE
        chim_line_gen = GenChimericReads(line_read1,line_read2,line_read12,left_line,right_line,'line',start)
        chim_hervk_gen = GenChimericReads(hervk_read1,hervk_read2,hervk_read12,left_hervk,right_hervk,'hervk',chim_line_gen.chim_read_count)
        chim_svaa_gen = GenChimericReads(svaa_read1,svaa_read2,svaa_read12,left_svaa,right_svaa,'svaa',chim_hervk_gen.chim_read_count)

        # Start genchimeric reads func for each object
        chim_line_gen.make_chim_reads()
        chim_hervk_gen.make_chim_reads()
        chim_svaa_gen.make_chim_reads()

        chim_line_gen.made_chimeric_reads.to_csv('made_chim_line.csv', sep='\t')
        chim_hervk_gen.made_chimeric_reads.to_csv('made_chim_hervk.csv', sep='\t')
        chim_svaa_gen.made_chimeric_reads.to_csv('made_chim_svaa.csv', sep='\t')

        # Use the read.finalize_read_and_write() function to send chimeric reads to temp fq
        #   which would then be randomized and finalized in the runner.py

        # Send line
        if chim_line_gen.in_read12_index_left_last > chim_line_gen.in_read12_index_right_last:
            start_iter = chim_line_gen.in_read12_index_left_last
        else:
            start_iter = chim_line_gen.in_read12_index_right_last

        _LOG.info('starting chim_lin_gen.in_read12.......................')
    
        for index,row in chim_line_gen.in_read12[start_iter:].iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)
            properly_paired_reads.append((row['read1'], row['read2']))
        _LOG.info('starting chim_lin_gen.made_chimeric_reads................')
        for index,row in chim_line_gen.made_chimeric_reads.iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)

            properly_paired_reads.append((row['read1'], row['read2']))

        # Send hervk
        if chim_hervk_gen.in_read12_index_left_last > chim_hervk_gen.in_read12_index_right_last:
            start_iter = chim_hervk_gen.in_read12_index_left_last
        else:
            start_iter = chim_hervk_gen.in_read12_index_right_last
        _LOG.info('starting chim_hervk_gen.in_read12...........................')
        for index,row in chim_hervk_gen.in_read12[start_iter:].iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)
            properly_paired_reads.append((row['read1'], row['read2']))
        _LOG.info('starting chim_hervk_gen.made_chimeric_reads...............................')
        for index,row in chim_hervk_gen.made_chimeric_reads.iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)
            properly_paired_reads.append((row['read1'], row['read2']))
    
        # Send svaa
        if chim_svaa_gen.in_read12_index_left_last > chim_svaa_gen.in_read12_index_right_last:
            start_iter = chim_svaa_gen.in_read12_index_left_last
        else:
            start_iter = chim_svaa_gen.in_read12_index_right_last
        _LOG.info('starting chim_svaa_gen.in_read12...................................')
        for index,row in chim_svaa_gen.in_read12[start_iter:].iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)
            properly_paired_reads.append((row['read1'], row['read2']))
        _LOG.info('starting chim_svaa_gen.made_chimeric_reads..................................')
        for index,row in chim_svaa_gen.made_chimeric_reads.iterrows():
            row['read1'].finalize_read_and_write(
                error_model_1, mutation_model, fq1_paired, options.quality_offset, options.produce_fastq
            )
            row['read2'].finalize_read_and_write(
                error_model_2, mutation_model, fq2_paired, options.quality_offset, options.produce_fastq
            )
            read1 = row['read1']
            _LOG.info(read1.name)
            properly_paired_reads.append((row['read1'], row['read2']))

    _LOG.info(f"Contig fastq(s) written in: {(time.time() - t)/60:.2f} m")

    if options.produce_bam:
        # this will give us the proper read order of the elements, for the sam. They are easier to sort now
        properly_paired_reads = sorted(properly_paired_reads)
        singletons = sorted(singletons)
        sam_order = properly_paired_reads + singletons

        with open_output(reads_pickle) as reads:
            pickle.dump(sam_order, reads)

        if options.paired_ended:
            _LOG.debug(f"Properly paired percentage = {len(properly_paired_reads)/len(sam_order)}")

    _LOG.info(f"Finished sampling reads in {(time.time() - start_time)/60:.2f} m")
    return chrom_fastq_r1_paired, chrom_fastq_r1_single, chrom_fastq_r2_paired, chrom_fastq_r2_single
