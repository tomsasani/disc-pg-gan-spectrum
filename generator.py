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
        reco_folder: str = None,
    ):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_haplotypes = sum(sample_sizes)
        self.rng = default_rng(seed)

        self.curr_params = None

        self.pretraining = False
        # for real data, use HapMap
        if reco_folder is not None:
            files = global_vars.get_reco_files(reco_folder)

            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)
        else: self.prior, self.weights = [], []

    def simulate_batch(
        self,
        # root_dists: np.ndarray,
        region_lens: np.ndarray,
        norm_len: int,
        batch_size: int = global_vars.BATCH_SIZE,
        params = [],
    ):
        """Simulate a batch of Generated regions. 

        Args:
            root_dists (np.ndarray): Root distributions to use for parameterizing the Generator.
                Should be a np.ndarray of shape (batch_size, 4).
            batch_size (int, optional): Batch size. Defaults to global_vars.BATCH_SIZE.
            params (list, optional): _description_. Defaults to [].
            real (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # initialize matrix in which to store data
        
        regions = np.zeros(
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
            dtype=np.float32)

        # set up parameters
        sim_params = param_set.ParamSet()
        if params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

        # simulate each region
        for i in range(batch_size):
            # set random seed
            seed = self.rng.integers(1, high=2**32)
            # simulate tree sequence using simulation parameters.
            # NOTE: the simulator simulates numbers of *samples* rather
            # than haplotypes, so we need to divide the sample sizes by 2
            # to get the correct number of haplotypes.
            ts = self.simulator(
                sim_params,
                [ss // 2 for ss in self.sample_sizes],
                # root_dists[i],
                region_lens[i],
                seed,
            )
            # return 3D array
            region, positions = prep_simulated_region(ts)
            assert region.shape[0] == positions.shape[0]
            #print (f"Simulated region is {positions}bp")
            region_formatted = util.process_region(region, positions, norm_len)
            regions[i] = region_formatted

        return regions

    def update_params(self, new_params):
        self.curr_params = new_params


def prep_simulated_region(ts) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    n_snps, n_haps = ts.genotype_matrix().astype(np.float32).shape

    # create the initial multi-dimensional feature array
    X = np.zeros((n_snps, n_haps))

    for var_idx, var in enumerate(ts.variants()):
        ref = var.alleles[0]
        alt_alleles = var.alleles[1:]
        gts = var.genotypes
        # ignore multi-allelics
        if len(alt_alleles) > 1: continue
        alt = alt_alleles[0]
        if ref in ("G", "T"):
            ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
        # shouldn't be any silent mutations given transition matrix, but make sure
        # we don't include them
        if ref == alt: continue
        mutation = ">".join([ref, alt])
        mutation_idx = global_vars.MUT2IDX[mutation]
        
        X[var_idx, :] = gts

    X = np.expand_dims(X, axis=2)
    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    seg = util.find_segregating_idxs(X)
    
    X_filtered = X[seg, :, :]
    
    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)
    assert positions.shape[0] == X.shape[0]

    filtered_positions = positions[seg]
    return X_filtered, filtered_positions

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
