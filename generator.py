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
        use_full_spectrum: bool = False,
    ):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_haplotypes = sum(sample_sizes)
        self.rng = default_rng(seed)
        self.use_full_spectrum = use_full_spectrum

        self.curr_params = None

        self.pretraining = False
        # for real data, use HapMap
        if reco_folder is not None:
            files = global_vars.get_reco_files(reco_folder)
            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)
        else: self.prior, self.weights = [], []

    def simulate_batch(
        self,
        root_dists: np.ndarray,
        batch_size: int = global_vars.BATCH_SIZE,
        params = [],
        treat_as_real: bool = False
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
        num_channels = 7 if self.use_full_spectrum else 2
        
        regions = np.zeros(
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, num_channels),
            dtype=np.float32)

        # set up parameters
        sim_params = param_set.ParamSet()
        if treat_as_real:
            pass
        elif params == []:
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
                self.sample_sizes,#[ss // 2 for ss in self.sample_sizes],
                root_dists[i],
                seed,
            )
            # return 3D array
            region, positions = prep_simulated_region(ts)
            assert region.shape[0] == positions.shape[0]
            region_formatted = util.process_region(region, positions, use_full_spectrum=self.use_full_spectrum)
            regions[i] = region_formatted

        return regions
    
    def real_batch(self, batch_size = global_vars.BATCH_SIZE):
        root_dists = np.tile(np.array([0.25] * 4), (batch_size, 1))
        return self.simulate_batch(root_dists, batch_size=batch_size, treat_as_real=True), root_dists

    def update_params(self, new_params):
        self.curr_params = new_params


def prep_simulated_region(ts) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    # the genotype matrix returned by tskit is our expected output
    # n_snps x n_haps matrix. however, even multi-allelic variants
    # are encoded 0/1, regardless of the allele.

    # the genotype matrix returned by tskit is our expected output
    # n_snps x n_haps matrix
    n_snps, n_haps = ts.genotype_matrix().astype(np.float32).shape
    X = np.zeros((n_snps, n_haps, 6), dtype=np.float32)

    for vi, var in enumerate(ts.variants()):
        ref = var.alleles[0]
        alt_alleles = var.alleles[1:]
        gts = var.genotypes
        # ignore multi-allelics
        assert len(alt_alleles) == 1

        alt = alt_alleles[0]
        if ref in ("G", "T"):
            ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
        # shouldn't be any silent mutations given transition matrix
        assert ref != alt
        mutation = ">".join([ref, alt])
        mutation_idx = global_vars.MUT2IDX[mutation]
        
        X[vi, :, mutation_idx] = gts

    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)
    assert positions.shape[0] == X.shape[0]

    # seg = util.find_segregating_idxs(X)
    # if seg.shape[0] < X.shape[0]:
    #     print (f"{seg.shape[0] / X.shape[0]} segregating sites in simulated data")
    #X_filtered = X[seg, :]

    #filtered_positions = positions[seg]
    return X, positions

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
