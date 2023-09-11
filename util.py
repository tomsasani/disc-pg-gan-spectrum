"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# our imports
import generator
import global_vars
import param_set
import real_data_random
import simulation

def inter_snp_distances(positions: np.ndarray, norm_len: int) -> np.ndarray:
    if positions.shape[0] > 0:
        dist_vec = [0] 
        for i in range(positions.shape[0] - 1):
            # NOTE: inter-snp distances always normalized to simulated region size
            dist_vec.append((positions[i + 1] - positions[i]) / norm_len)
    else: dist_vec = []
    return np.array(dist_vec)

def sort_by_genetic_similarity():
    pass

def sum_across_channels(X: np.ndarray) -> np.ndarray:
    summed = np.sum(X, axis=2)
    #assert np.max(summed) == 1
    return np.expand_dims(summed, axis=2)

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
        derived_sum_rescaled = derived_sum / channel_maxes.reshape(1, -1)
        derived_sum_rescaled[np.isnan(derived_sum_rescaled)] = 0.
        rel_regions_sum[:, i, :] = derived_sum_rescaled
    return rel_regions_sum


def reorder(X: np.ndarray):
    return np.transpose(X, (1, 0, 2))


def process_region(
    X: np.ndarray,
    positions: np.ndarray,
    norm_len: int,
) -> np.ndarray:
    """
    Process an array of shape (n_sites, n_haps, 6), which is produced
    from either generated or real data. First, subset it to contain global_vars.NUM_SNPS
    polymorphisms, and then calculate the sums of derived alleles on each haplotype in global_vars.N_WINDOWS
    windows across the arrays. 
    
    Zero-pad if necessary.

    Args:
        X (np.ndarray): feature array of shape (n_sites, n_haps, n_channels - 1)
        positions (np.ndarray): array of positions that correspond to each of the
            n_sites in the feature array.

    Returns:
        np.ndarray: _description_
    """
    # figure out how many sites and haplotypes are in the actual
    # multi-dimensional array
    n_sites, n_haps = X.shape
    # make sure we have exactly as many positions as there are sites
    assert n_sites == positions.shape[0]
    # get inter-SNP distances, relative to the simualted region size
    distances = inter_snp_distances(positions, norm_len)

    #print (f"Processing distances with region length of {norm_len}")

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = n_sites // 2

    half_S = global_vars.NUM_SNPS // 2

    # instantiate the new region, formatted as (n_haps, n_sites, n_channels)
    region = np.zeros(
        (n_haps, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
        dtype=np.float32,
    )

    # if we have more than the necessary number of SNPs
    if mid >= half_S:
        # add first channels of mutation spectra
        middle_X = np.transpose(X[mid - half_S:mid + half_S, :], (1, 0))
        region[:, :, :-1] = np.expand_dims(major_minor(middle_X), axis=2)
        # tile the inter-snp distances down the haplotypes
        distances_tiled = np.tile(distances[mid - half_S:mid + half_S], (n_haps, 1))
        # add final channel of inter-snp distances
        region[:, :, -1] = distances_tiled

    else:
        other_half_S = half_S + 1 if n_sites % 2 == 1 else half_S
        # use the complete genotype array
        # but just add it to the center of the main array
        region[:, half_S - mid:mid + other_half_S, :-1] = np.expand_dims(major_minor(np.transpose(X, (1, 0))), axis=2)
        # tile the inter-snp distances down the haplotypes
        distances_tiled = np.tile(distances, (n_haps, 1))
        # add final channel of inter-snp distances
        region[:, half_S - mid:mid + other_half_S, -1] = distances_tiled

    # convert anc/der alleles to -1, 1
    return region

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


def major_minor(matrix):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""

    # NOTE: need to fix potential mispolarization if using ancestral genome?
    n_haps, n_sites = matrix.shape
    for site_i in range(n_sites):
        #for mut_i in range(n_channels):
            # if greater than 50% of haplotypes are ALT, reverse
            # the REF/ALT polarization
        if np.count_nonzero(matrix[:, site_i] > 0) > (n_haps / 2):
            matrix[:, site_i] = 1 - matrix[:, site_i]
    # option to convert from 0/1 to -1/+1
    matrix[matrix == 0] = -1

    # matrix[matrix > 1] = 1
    return matrix


def parse_args():
    """Parse command line arguments."""

    p = argparse.ArgumentParser()

    p.add_argument(
        '--data',
        type=str,
        help='real data file in hdf5 format',
        required=True,
    )
    p.add_argument(
        '--ref',
        type=str,
        help="path to ancestral reference",
        required=True,
    )

    p.add_argument(
        '--disc',
        type=str,
        dest='disc',
        help='location to store discriminator',
    )
    p.add_argument(
        '-bed',
        type=str,
        help='bed file (mask)',
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


def process_args(args):

    # parameter defaults
    parameters = parse_params(args.params) # desired params
    param_names = [p.name for p in parameters]

    # initialize the Iterator object, which will iterate over
    # the VCF and grab regions with NUM_SNPS
    iterator = real_data_random.RealDataRandomIterator(
        hdf_fh=args.data,
        ref_fh=args.ref,
        bed_file=args.bed,
        seed=args.seed,
    )
    # figure out how many haplotypes are being sampled by the Iterator from the VCF
    h_total = iterator.num_haplotypes

    print (f"ITERATOR is using {h_total} haplotypes")

    # always using the EXP model, with a single population
    num_pops = 1
    simulator = simulation.simulate_isolated

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
