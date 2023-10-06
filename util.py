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
from sklearn.neighbors import NearestNeighbors

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


def sort_min_diff(X):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''
    # reduce to 2 dims
    X = X[:, :, 0]
    mb = NearestNeighbors(n_neighbors=len(X), metric='manhattan').fit(X)
    v = mb.kneighbors(X)
    smallest = np.argmin(v[0].sum(axis=1))
    return v[1][smallest]
    #return X[v[1][smallest]]

def parse_hapmap_empirical_prior(files):
    """
    Parse recombination maps to create a distribution of recombintion rates to
    use for real data simulations. Based on defiNETti software package.
    """
    print("Parsing HapMap recombination rates...")

    # set up weights (probabilities) and reco rates
    weights_all = []
    prior_rates_all = []

    for f in files:
        mat = np.loadtxt(f, skiprows = 1, usecols=(1,2))
        #print(mat.shape)
        mat[:,1] = mat[:,1]*(1.e-8)
        mat = mat[mat[:,1] != 0.0, :] # remove 0s
        weights = mat[1:,0] - mat[:-1,0]
        prior_rates = mat[:-1,1]

        weights_all.extend(weights)
        prior_rates_all.extend(prior_rates)

    # normalize
    prob = weights_all / np.sum(weights_all)

    # make smaller by a factor of 50 (collapse)
    indexes = list(range(len(prior_rates_all)))
    indexes.sort(key=prior_rates_all.__getitem__)

    prior_rates_all = [prior_rates_all[i] for i in indexes]
    prob = [prob[i] for i in indexes]

    new_rates = []
    new_weights = []

    collapse = 50
    for i in range(0,len(prior_rates_all),collapse):
        end = collapse
        if len(prior_rates_all)-i < collapse:
            end = len(prior_rates_all)-i
        new_rates.append(sum(prior_rates_all[i:i+end])/end) # average
        new_weights.append(sum(prob[i:i+end])) # sum

    new_rates = np.array(new_rates)
    new_weights = np.array(new_weights)

    return new_rates, new_weights

def find_segregating_idxs(X: np.ndarray):

    n_snps, n_haps = X.shape
    # initialize mask to store "good" sites
    to_keep = np.ones(n_snps)
    # find sites where there are bi-allelics
    multi_allelic = np.where(np.any(X > 1, axis=1))[0]
    if multi_allelic.shape[0] > 0: print ("found multi-allelic")
    to_keep[multi_allelic] = 0
    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    acs = np.count_nonzero(X, axis=1)
    non_segregating = np.where((acs == 0) | (acs == n_haps))[0]
    if non_segregating.shape[0] > 0: print ("found non seg")
    to_keep[non_segregating] = 0
    return np.where(to_keep)[0]


def process_region(
    X: np.ndarray,
    positions: np.ndarray,
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

    # check for multi-allelics
    #assert np.max(X) == 1

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = n_sites // 2
    half_S = global_vars.NUM_SNPS // 2

    # instantiate the new region, formatted as (n_haps, n_sites, n_channels)
    region = np.zeros(
        (n_haps, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
        dtype=np.float32,
    )

    # should we divide by the *actual* region length?
    distances = inter_snp_distances(positions, global_vars.L)

    # first, transpose the full input matrix to be n_haps x n_snps
    X = X.T

    # if we have more than the necessary number of SNPs
    if mid >= half_S:
        # define indices to use for slicing
        i, j = mid - half_S, mid + half_S
        # add sites to output
        region[:, :, 0] = major_minor(X[:, i:j])
        # add one-hot to output
        # tile the inter-snp distances down the haplotypes
        # get inter-SNP distances, relative to the simualted region size
        distances_tiled = np.tile(distances[i:j], (n_haps, 1))
        # add final channel of inter-snp distances
        region[:, :, -1] = distances_tiled

    else:
        other_half_S = half_S + 1 if n_sites % 2 == 1 else half_S
        i, j = half_S - mid, mid + other_half_S
        # use the complete genotype array
        # but just add it to the center of the main array
        region[:, i:j, 0] = major_minor(X)
        # add one-hot to output
        # tile the inter-snp distances down the haplotypes
        distances_tiled = np.tile(distances, (n_haps, 1))
        # add final channel of inter-snp distances
        region[:, i:j, -1] = distances_tiled

    # np.random.shuffle(region)

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

    # figure out the channel in which each mutation occurred
    for site_i in range(n_sites):
        # in this channel, figure out whether this site has any
        # derived alleles
        haplotypes = matrix[:, site_i]
        # if not, we'll mask all haplotypes at this site on this channel,
        # leaving the channel with the actual mutation unmasked
        if np.sum(haplotypes) > (n_haps / 2):
            # if greater than 50% of haplotypes are ALT, reverse
            # the REF/ALT polarization
            haplotypes = 1 - haplotypes

        matrix[:, site_i] = haplotypes

    matrix[matrix == 0] = -1

    return matrix


def parse_args():
    """Parse command line arguments."""

    p = argparse.ArgumentParser()

    p.add_argument(
        '-data',
        type=str,
        help='real data file in hdf5 format',
    )
    p.add_argument(
        '-ref',
        type=str,
        help="path to ancestral reference",
    )

    p.add_argument(
        '-disc',
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
    p.add_argument(
        '-outpref',
        type=str,
        default="out",
        help="prefix for output csv",
    )
    p.add_argument(
        "-spectrum",
        action="store_true",
        help="whether to use a 7-channel mutation spectrum image instead of a 2-channel mutation image",
    )
    p.add_argument(
        "-entropy",
        default="n",
        type=str,
    )

    args = p.parse_args()

    return args


def process_args(args):

    # parameter defaults
    parameters = parse_params(args.params) # desired params
    param_names = [p.name for p in parameters]

    # always using the EXP model, with a single population
    num_pops = 2
    simulator = simulation.simulate_im

    # if a data path is specified, we'll use those data for
    # the Iterator object.
    if args.data is not None:
        # initialize the Iterator object, which will iterate over
        # the h5 file and grab regions with NUM_SNPS
        iterator = real_data_random.RealDataRandomIterator(
            hdf_fh=args.data,
            bed_file=args.bed,
            seed=args.seed,
        )
        h_total = iterator.num_haplotypes
        sample_sizes = [h_total // num_pops for _ in range(num_pops)]
        #sample_sizes = [14, 8]

    # otherwise, we'll use the Generator
    else:
        sample_sizes = [global_vars.NUM_HAPLOTYPES // num_pops for _ in range(num_pops)]
        iterator = generator.Generator(
            simulator=simulator,
            param_names=param_names,
            sample_sizes=sample_sizes,
            #sample_sizes = sample_sizes,
            seed=args.seed,
        )
        h_total = global_vars.NUM_HAPLOTYPES


    print (f"ITERATOR is using {h_total} haplotypes")

    # figure out how many haplotypes the Generator object should be simulating in each batch
    # we want this to match the number of haplotypes being sampled in the VCF
    #sample_size_total = ss_total if args.sample_size is None else args.sample_size
    print (f"GENERATOR is simulating {sum(sample_sizes)} haplotypes")

    gen = generator.Generator(
        simulator=simulator,
        param_names=param_names,
        sample_sizes=sample_sizes,
        seed=args.seed,
    )

    return gen, iterator, parameters, sample_sizes, args.entropy

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
