"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from numpy.random import default_rng

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
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, 6),
            dtype=np.float32)

        # set up parameters
        sim_params = param_set.ParamSet()
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

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
                seed,
            )
            # return 3D array
            region = prep_region(ts, neg1)
            region_formatted = util.process_region(region)
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


def prep_region(ts, neg1=False) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    n_sites, n_haps = ts.genotype_matrix().astype(np.float32).shape

    X = np.zeros((n_sites, n_haps, 6))
    var_idx = 0
    for var in ts.variants():
        ref = var.alleles[0]
        alts = var.alleles[1:]
        pos, gts = int(var.site.position), var.genotypes
        # if ref not in revcomp.keys():
        #     continue
        for alt_idx, alt in enumerate(alts):
            haps_with_alt = np.where(gts == alt_idx + 1)[0]
            if ref in ("G", "T"):
                ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
            # TODO: deal with silent mutations
            if ref == alt: continue
            mutation = ">".join([ref, alt])
            mutation_idx = global_vars.MUT2IDX[mutation]
            X[var_idx, haps_with_alt, mutation_idx] += 1
        var_idx += 1
    return X

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
