"""
This function parses the various bed files that can be input into NEAT. It has a special subtask if the input
is a mutation rate dictionary, which will parse out the input mutation rates associated with each region.
"""
import pathlib
import logging
import numpy as np

from Bio.File import _IndexedSeqFileDict

from ...common import open_input
from .options import Options

__all__ = [
    "parse_beds",
    "fill_out_bed_dict"
]

_LOG = logging.getLogger(__name__)


def parse_beds(options: Options, reference_dict: _IndexedSeqFileDict, average_mutation_rate: float) -> list:
    """
    This single function parses the three possible bed file types for NEAT.

    :param options: The options object for this run
    :param reference_dict: The reference dict generated by SeqIO for this run
    :param average_mutation_rate: The average mutation rate from the model or user input for this run
    :return: target_dict, discard_dict, mutation_rate_dict
    """

    return_list = []
    bed_files = (options.target_bed, options.discard_bed, options.mutation_bed)
    # These numbers indicate the factors for target_bed, discard_bed, and mutation_bed, respectively
    factors = {0: 1, 1: 0, 2: average_mutation_rate}

    for i in range(len(bed_files)):
        # To distinguish between a targeted bed with chromosomes intentionally omitted and no target bed at all
        targeted = True if i == 0 else False
        temp_bed_dict = parse_single_bed(
            bed_files[i],
            reference_dict,
            targeted
        )

        final_dict = fill_out_bed_dict(reference_dict, temp_bed_dict, factors[i], average_mutation_rate, options)

        return_list.append(final_dict)

    return return_list


def parse_single_bed(input_bed: str,
                     reference_dictionary: _IndexedSeqFileDict,
                     targeted_bed: bool
                     ) -> dict:
    """
    Will parse a bed file, returning a dataframe of chromosomes that are found in the reference,
    and the corresponding positions. Some beds may have mutation in the fourth column. If specified
    then we will also import the mutation data.

    :param input_bed: A bed file containing the regions of interest
    :param reference_dictionary: A list of chromosomes to check, along with their reads and lengths
    :param targeted_bed: Indicates that we are processing the targeted bed, which requires one additional factor,
        to indicate if a chromosome was left out of the targeted bed intentionally or the targeted bed was never present.
    :return: a dictionary of chromosomes: [(pos1, pos2, factor), etc...]
    """
    ret_dict = {x: [] for x in reference_dictionary}
    in_bed_only = []

    if input_bed:
        # Pathlib will help us stay machine-agnostic to the degree possible
        input_bed = pathlib.Path(input_bed)
        printed_chromosome_warning = False
        printed_mutation_rate_warning = False
        with open_input(input_bed) as f:
            for line in f:
                if not line.startswith(('@', '#', "\n")):
                    # Note: on targeted and discard regions, we really only need chrom, pos1, and pos2
                    # But for the mutation rate regions, we need a fourth column of meta_data,
                    # So we have to check them all, though it won't be used for targeted and discard regions
                    line_list = line.strip().split('\t')[:4]
                    try:
                        [my_chr, pos1, pos2] = line_list[:3]
                    except ValueError:
                        _LOG.error(f"Improperly formatted bed file line {line}")
                        raise
                    # Trying not to 'fix' bed files, but an easy case is if the vcf uses 'chr' and the bed doesn't, in
                    # which we frequently see in human data, we'll just put chr on there. Anything more complicated
                    # will be on the user to correct their beds first.
                    try:
                        assert my_chr in reference_dictionary
                    except AssertionError:
                        try:
                            assert my_chr in reference_dictionary
                        except AssertionError:
                            in_bed_only.append(my_chr)
                            if not printed_chromosome_warning:
                                _LOG.warning("Found chromosome in BED file that isn't in Reference file, skipping")
                                printed_chromosome_warning = True
                            continue

                    if len(line_list) > 3:
                        # here we append the metadata, if present
                        index = line_list[3].find('mut_rate=')
                        if index == -1:
                            if not printed_mutation_rate_warning:
                                _LOG.warning(f"Found no mutation rates in bed")
                                _LOG.warning(f'4th column of mutation rate bed must be a semicolon list of key, value '
                                           f'pairs, with one key being mut_rate, e.g., "foo=bar;mut_rate=0.001;do=re".')
                                printed_mutation_rate_warning = True
                            continue

                        # +9 because that's len('mut_rate='). Whatever is that should be our mutation rate.
                        mut_rate = line_list[3][index + 9:]
                        # We'll trim anything after the mutation rate and call it good. These should be ; separated
                        try:
                            mut_rate = float(mut_rate.split(';')[0])
                        except ValueError:
                            _LOG.error(f"Invalid mutation rate: {my_chr}: ({pos1}, {pos2})")
                            _LOG.debug(f'4th column of mutation rate bed must be a semicolon list of key, value '
                                       f'pairs, with one key being mut_rate, e.g., "foo=bar;mut_rate=0.001;do=re".')
                            raise

                        if mut_rate > 0.3:
                            _LOG.warning("Found a mutation rate > 0.3. This is unusual.")

                        ret_dict[my_chr].append((int(pos1), int(pos2), mut_rate))
                    else:
                        ret_dict[my_chr].append((int(pos1), int(pos2)))

        # some validation
        in_ref_only = [k for k in reference_dictionary if k not in ret_dict]
        if in_ref_only:
            _LOG.warning(f'Warning: Reference contains sequences not found in BED file {input_bed}. '
                         f'These chromosomes will be omitted from the outputs.')
            _LOG.debug(f"In reference only regions: {in_ref_only}")

        if in_bed_only:
            _LOG.warning(f'BED file {input_bed} contains sequence names '
                         f'not found in reference. These regions will be ignored.')
            _LOG.debug(f'Regions ignored: {list(set(in_bed_only))}')

    elif targeted_bed:
        # This is to indicate that there was no targeted bed. Otherwise the code will try to assign a 2% coverage
        # rate to the entire dataset
        ret_dict = {x: False for x in reference_dictionary}

    return ret_dict


def fill_out_bed_dict(ref_dict: _IndexedSeqFileDict,
                      region_dict: dict | None,
                      factor: float | int,
                      avg_mut_rate: float,
                      options: Options,
                      ) -> dict:
    """
    This parses the dict derived from the bed file and fills in any gaps, so it can be more easily cycled through
    later.

    The input to this function is the dict for a single chromosome.

    :param ref_dict: The reference dictionary for the run. Needed to calulate max values. We shouldn't need to access the data it
        refers to.
    :param region_dict: A dict based on the input from the bed file, with keys being all the chronmosomes
        present in the reference.
    :param factor: Depending on the bed type, this will either be a 1 (for targeted regions), 0 (for discarded regions),
        or the average mutation rate desired for this dataset. If not present, then this function
        will only fill out the regions, not try to assign values to them
    :param avg_mut_rate: This is the average mutation rate for the run.
    :param options: options for this run
    :return: A tuple with (start, end, some_factor) for each region in the genome.
    """

    ret_dict = {}
    if factor == 1:
        other_factor = options.off_target_scalar
    elif factor == 0:
        other_factor = 1
    else:
        other_factor = avg_mut_rate

    for contig in region_dict:
        regions = []
        max_value = len(ref_dict[contig])
        start = 0

        # If an input bed was supplied, this will set up the regions for it
        if region_dict[contig]:
            for region in region_dict[contig]:
                if len(region) > 2:
                    factor = region[2]
                if region[0] > start:
                    regions.append((start, region[0], other_factor))
                    start = region[1]
                    regions.append((region[0], region[1], factor))
                elif region[0] == start:
                    regions.append((region[0], region[1], factor))
                    start = region[1]

            # If the region ends short of the end, this fills in to the end
            if regions[-1][1] != max_value:
                regions.append((start, max_value, other_factor))

            ret_dict[contig] = regions

        # If no input bed was supplied, then we check if the datatype is boolean, in which case we know that the
        # bed in question was the targeted bed and since the above condition was false, there was no targeted bed
        # supplied, so we want full coverage on all contigs.
        # This is to prevent the coverage from being 2% of the target when no targeted bed file was included.

        elif type(region_dict[contig]) == bool:
            ret_dict[contig] = [(0, max_value, 1)]

        # In all other cases, the value is the other_factor
        else:
            ret_dict[contig] = [(0, max_value, other_factor)]

    return ret_dict
