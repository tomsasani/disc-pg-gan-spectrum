"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
import seaborn as sns
# our imports
import global_vars
import param_set
import simulation
import util

################################################################################
# GENERATOR CLASS
################################################################################

class Generator:

    def __init__(
        self,
        simulator,
        param_names,
        sample_sizes,
        seed,
    ):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_haplotypes = sum(sample_sizes)
        self.rng = default_rng(seed)
        self.curr_params = None

        self.pretraining = False
        self.prior, self.weights = [], []

    def simulate_batch(
        self,
        root_dists: np.ndarray,
        region_lens: np.ndarray,
        batch_size: int = global_vars.BATCH_SIZE,
        params = [],
        real: bool = False,
        neg1: bool = True,
    ):
        """Simulate a batch of Generated regions. 

        Args:
            root_dists (np.ndarray): Root distributions to use for parameterizing the Generator.
                Should be a np.ndarray of shape (batch_size, 4).
            batch_size (int, optional): Batch size. Defaults to global_vars.BATCH_SIZE.
            params (list, optional): _description_. Defaults to [].
            real (bool, optional): _description_. Defaults to False.
            neg1 (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        # initialize matrix in which to store data
        regions = np.zeros(
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
            dtype=np.float32)

        # set up parameters
        sim_params = param_set.ParamSet()
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

        #print (f"""Params used for generator loss: N1 = {sim_params.N1.value}, N2 = {sim_params.N2.value}, T1 = {sim_params.T1.value}, T2 = {sim_params.T2.value}, mu = {sim_params.mu.value}, rho = {sim_params.rho.value}""")

        # simulate each region
        for i in range(batch_size):
            # set random seed
            seed = self.rng.integers(1, high=2**32,)
            # simulate tree sequence using simulation parameters.
            # NOTE: the simulator simulates numbers of *samples* rather
            # than haplotypes, so we need to divide the sample sizes by 2
            # to get the correct number of haplotypes.
            #print (i, root_dists[i])
            ts = self.simulator(
                sim_params,
                [ss // 2 for ss in self.sample_sizes],
                root_dists[i],
                region_lens[i],
                seed,
            )
            # return 3D array
            region, dist_vec = prep_region(ts)
            region_formatted = util.process_region(region, dist_vec, neg1=neg1)
            regions[i] = region_formatted

        return regions

    def real_batch(
        self,
        root_dists: np.ndarray,
        batch_size=global_vars.BATCH_SIZE,
        neg1=True,
        region_len=False,
    ):
        return self.simulate_batch(root_dists, batch_size=batch_size, real=True, neg1=neg1,
            region_len=region_len)

    def update_params(self, new_params):
        self.curr_params = new_params


def prep_region(ts) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    n_sites, n_haps = ts.genotype_matrix().astype(np.float32).shape

    # create the initial multi-dimensional feature array
    X = np.zeros((n_sites, n_haps, 6))
    for var_idx, var in enumerate(ts.variants()):
        ref = var.alleles[0]
        alts = var.alleles[1:]
        gts = var.genotypes
        # ignore multi-allelics
        if len(alts) > 1: 
            #print ("Found a multi-allelic!")
            continue
        for alt_idx, alt in enumerate(alts):
            haps_with_alt = np.where(gts == alt_idx + 1)[0]
            if ref in ("G", "T"):
                ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
            # shouldn't be any silent mutations given transition matrix, but make sure
            # we don't include them
            if ref == alt: 
                #print ("Encountered a silent mutation!")
                continue
            mutation = ">".join([ref, alt])
            mutation_idx = global_vars.MUT2IDX[mutation]
            
            X[var_idx, haps_with_alt, mutation_idx] += 1

    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    summed_across_channels = np.sum(X, axis=2)
    summed_across_haplotypes = np.sum(summed_across_channels, axis=1)
    seg = np.where((summed_across_haplotypes > 0) & (summed_across_haplotypes < n_haps))[0]
    # if seg.shape[0] < n_sites:
    #    print (f"Found {n_sites - seg.shape[0]} non-segregating sites in the fake data.")
    X = X[seg, :, :]
    # NOTE: we don't apply a strict filter on region size when using HDF5
    # format for storing genotypes, so our inter-snp distances will always
    # be a function of the sampled region size (i.e., the largest position)
    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)
    filtered_positions = positions[seg]
    # create vector of inter-SNP distances
    # TODO: hacky
    if filtered_positions.shape[0] > 0:
        dist_vec = [0] 
        region_len = np.max(filtered_positions) - np.min(filtered_positions)
        for i in range(filtered_positions.shape[0] - 1):
            dist_vec.append((filtered_positions[i + 1] - filtered_positions[i]) / region_len)
    else: dist_vec = []
    return X, np.array(dist_vec)

# testing
if __name__ == "__main__":

    batch_size = 10
    params = param_set.ParamSet()

    # quick test
    print("sim exp")
    generator = Generator(simulation.simulate_exp, ["N1", "kappa", "conversion", "mu"], [20],
                          global_vars.DEFAULT_SEED)
    generator.update_params([params.N1.value, params.kappa.value, params.conversion.value, params.mu.value,])

    test_root_dists = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (batch_size, 1))

    mini_batch = generator.simulate_batch(
        test_root_dists,
        batch_size=batch_size,
    )

    print("x", mini_batch.shape)
