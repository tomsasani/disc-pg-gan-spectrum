"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import argparse
import sys

# our imports
import generator
import global_vars
import param_set
import real_data_random
import simulation

def sum_across_windows(X: np.ndarray) -> np.ndarray:

    n_haps, n_sites, n_channels = X.shape
    window_size = global_vars.WINDOW_SIZE
    windows = np.arange(0, global_vars.NUM_SNPS, step=window_size)
    window_starts, window_ends = windows[:-1], windows[1:]

    rel_regions_sum = np.zeros((n_haps, window_starts.shape[0], n_channels))
    for i, (s, e) in enumerate(zip(window_starts, window_ends)):
        # get count of derived alleles in window
        derived_sum = np.nansum(X[:, s:e, :], axis=1)
        channel_maxes = np.max(derived_sum, axis=0)
        print (derived_sum)
        derived_sum_rescaled = derived_sum / channel_maxes.reshape(1, -1)
        derived_sum_rescaled[np.isnan(derived_sum_rescaled)] = 0.
        print (derived_sum_rescaled)
        rel_regions_sum[:, i, :] = derived_sum_rescaled
    return rel_regions_sum

def process_region(X: np.ndarray, neg1: bool = True) -> np.ndarray:
    """
    Process an array of shape (n_sites, n_haps, 6), which is produced
    from either generated or real data. First, subset it to contain global_vars.NUM_SNPS
    polymorphisms, and then calculate the sums of derived alleles on each haplotype in global_vars.N_WINDOWS
    windows across the arrays. 
    
    Zero-pad if necessary.

    Args:
        X (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    n_sites, n_haps = X.shape[:-1]

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = n_sites // 2

    S = global_vars.NUM_SNPS

    half_S = S // 2
    other_half_S = half_S

    # instantiate the new region, formatted as (n_haps, n_sites, 6)
    region = np.zeros((n_haps, global_vars.NUM_SNPS, 6), dtype=np.float32)

    # enough SNPs, take middle portion
    if mid >= half_S:
        middle_portion = X[mid - half_S:mid + other_half_S, :, :]
        # compute sums of derived alleles on each haplotype for each mutation type
        #middle_portion_sums = compute_haplotype_sums(middle_portion, window_size = global_vars.NUM_SNPS)
        # transpose so that haplotypes (not sites) are in 0th dimension
        #middle_portion_sums = np.transpose(middle_portion_sums, (1, 0, 2))
        #print (middle_portion_sums.shape)
        region[:, :, :] = np.transpose(middle_portion, (1, 0, 2))

    else:
        #print("NOT ENOUGH SNPS", n_sites)
        if n_sites % 2 == 1: other_half_S += 1
        # use the complete genotype array
        # but just add it to the center of the main array
        region[:, half_S - mid:mid + other_half_S, :] = np.transpose(
                   X, (1, 0, 2))
    #print (region)
    #print (sum_across_windows(region))
    # convert anc/der alleles to -1, 1
    return sum_across_windows(region)
    #return major_minor(region, neg1)

def parse_params(param_input):
    """See which params were desired for inference"""
    param_strs = param_input.split(',')
    parameters = []
    for _, p in vars(param_set.ParamSet()).items():
        if p.name in param_strs:
            parameters.append(p)

    assert len(parameters) == len(param_strs)
    for p in parameters:
        print(p)

    return parameters


def major_minor(matrix, neg1):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        if np.count_nonzero(matrix[:,j] > 0) > (n/2): # count the 1's
            matrix[:,j] = 1 - matrix[:,j]

    # option to convert from 0/1 to -1/+1
    if neg1:
        matrix[matrix == 0] = -1
    # residual numbers higher than one may remain even though we restricted to
    # biallelic
    #matrix[matrix > 1] = 1 # removing since we filter in VCF
    return matrix

def prep_real(gt, snp_start, snp_end, indv_start, indv_end):
    """Slice out desired region and unravel haplotypes"""
    region = gt[snp_start:snp_end, indv_start:indv_end, :]
    both_haps = np.concatenate((region[:,:,0], region[:,:,1]), axis=1)
    return both_haps

def filter_nonseg(region):
    """Filter out non-segregating sites in this region"""
    nonseg0 = np.all(region == 0, axis=1) # row all 0
    nonseg1 = np.all(region == 1, axis=1) # row all 1
    keep0 = np.invert(nonseg0)
    keep1 = np.invert(nonseg1)
    filter = np.logical_and(keep0, keep1)
    return filter

def parse_args():
    """Parse command line arguments."""

    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_vcf',
        type=str,
        help='real data file in VCF format',
        required=True,
    )
    p.add_argument(
        '--ref',
        type=str,
        help="path to ancestral reference",
        required=True,
    )
    p.add_argument(
        '--bed',
        type=str,
        help='bed file (mask)',
        required=True,
    )
    p.add_argument(
        '--disc',
        type=str,
        dest='disc',
        help='location to store discriminator',
    )
    p.add_argument(
        '-params',
        type=str,
        help='comma separated parameter list',
        default="N1,N2,T1,T2,rho,mu,growth,kappa,conversion,conversion_length"
    )
    p.add_argument(
        '-reco_folder',
        type=str,
        help='recombination maps',
    )
    p.add_argument(
        '-toy',
        action="store_true",
        help='toy example',
    )
    p.add_argument(
        '-seed',
        type=int,
        default=1833,
        help='seed for RNG',
    )

    args = p.parse_args()

    return args


def process_args(args, summary_stats = False):

    # parameter defaults
    all_params = param_set.ParamSet()
    parameters = parse_params(args.params) # desired params
    param_names = [p.name for p in parameters]

    real = False
    #ss_total = global_vars.DEFAULT_SAMPLE_SIZE

    # initialize the Iterator object, which will iterate over
    # the VCF and grab regions of the size specified in global_vars.L
    iterator = real_data_random.RealDataRandomIterator(
        vcf_fh=args.data_vcf,
        ref_fh=args.ref,
        bed_file=args.bed,
        seed=args.seed,
    )
    # figure out how many haplotypes are being sampled by the Iterator from the VCF
    h_total = iterator.num_haplotypes

    print (f"ITERATOR is using {h_total} haplotypes")

    # always using the EXP model, with a single population
    num_pops = 1
    simulator = simulation.simulate_exp

    if (global_vars.FILTER_SIMULATED or global_vars.FILTER_REAL_DATA):
        print("FILTERING SINGLETONS")

    # figure out how many haplotypes the Generator object should be simulating in each batch
    # we want this to match the number of haplotypes being sampled in the VCF
    #sample_size_total = ss_total if args.sample_size is None else args.sample_size
    sample_sizes = [h_total // num_pops for _ in range(num_pops)]

    print (f"GENERATOR is simulating {sample_sizes[0]} haplotypes")

    gen = generator.Generator(
        simulator,
        param_names,
        sample_sizes,
        args.seed,
    )

    return gen, iterator, parameters, sample_sizes

if __name__ == "__main__":
    # test major/minor and post-processing
    global_vars.NUM_SNPS = 4 # make smaller for testing

    a = np.zeros((6,3))
    a[0,0] = 1
    a[1,0] = 1
    a[2,0] = 1
    a[3,0] = 1
    a[0,1] = 1
    a[1,1] = 1
    a[2,1] = 1
    a[4,2] = 1
    dist_vec = [0.3, 0.2, 0.4, 0.5, 0.1, 0.2]

    print(a)
    print(major_minor(a, neg1=True))

    process_region(a, dist_vec, real=False)
