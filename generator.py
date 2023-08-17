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
        mirror_real=False,
        reco_folder="",
    ):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes)
        self.rng = default_rng(seed)
        self.curr_params = None

        self.pretraining = False
        self.prior, self.weights = [], []

        # for real data, use HapMap
        # if mirror_real and reco_folder != None:
        #     files = global_vars.get_reco_files(reco_folder)
        #     self.prior, self.weights = util.parse_hapmap_empirical_prior(files)



    def simulate_batch(
        self,
        batch_size=global_vars.BATCH_SIZE,
        params=[],
        sample_sizes=[],
        region_len=False,
        real=False,
        neg1=True,
    ):

        # initialize 4D matrix (batch, num_samples, num_snps, num_mutation_types)
        if region_len:
            regions = []
        else:
            regions = np.zeros(
                (batch_size, self.num_samples, global_vars.NUM_SNPS, 6),
                dtype=np.float32,
            )

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
            print (sim_params.N1.value)
            # set random seed
            seed = self.rng.integers(1, high=2**32)
            # simulate tree sequence using simulation parameters
            ts = self.simulator(
                sim_params,
                self.sample_sizes,
                seed,
            )
            # return 3D array
            region = prep_region(ts)#, neg1, region_len=region_len)
                
            while region is None:
                seed = self.rng.integers(1, high=2**32)
                ts = self.simulator(sim_params, self.sample_sizes, seed)
                region = prep_region(ts)

            if region_len:
                regions.append(region)
            else:
                regions[i] = region

        return regions

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
        region_len=False):
        return self.simulate_batch(batch_size=batch_size, real=True, neg1=neg1,
            region_len=region_len)

    def update_params(self, new_params):
        self.curr_params = new_params

    def get_reco(self, params):
        if self.prior == []:
            return params.reco.value

        return draw_background_rate_from_prior(self.prior, self.weights, self.rng)

def draw_background_rate_from_prior(prior_rates, prob, rng):
    return rng.choice(prior_rates, p=prob)


def prep_region(ts, neg1=False) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    revcomp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    mut2idx = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
    # keep track of mutations with actual mutation changes
    n_sites = len([var for var in ts.variants() if var.alleles[0] != "0"])

    n_haps = ts.genotype_matrix().astype(np.int8).shape[1]
    # get middle NUM_SITES
    if n_sites < global_vars.NUM_SNPS:
        return None
    else:
        X = np.zeros((n_sites, n_haps, 6))
        var_idx = 0
        for var in ts.variants():
            ref = var.alleles[0]
            alts = var.alleles[1:]
            pos, gts = int(var.site.position), var.genotypes
            # NOTE: what is going on with mutation changes from 0 > 1?
            # only occurring at "float" positions
            if ref not in revcomp.keys(): 
                #print (var.site.position, var.alleles, var.genotypes)
                continue
            for alt_idx, alt in enumerate(alts):
                haps_with_alt = np.where(gts == alt_idx + 1)[0]
                #print (alt_idx, alt, haps_with_alt)
                if ref in ("G", "T"):
                    ref, alt = revcomp[ref], revcomp[alt]
                mutation = ">".join([ref, alt])
                mutation_idx = mut2idx[mutation]
                X[var_idx, haps_with_alt, mutation_idx] += 1
            var_idx += 1

        mid = n_sites // 2
        S = global_vars.NUM_SNPS
        half_S = S // 2
        other_half_S = half_S 
        # if n_sites % 2 == 1: # odd
        #     other_half_S = half_S + 1
        # else:
        #     other_half_S = half_S
        # set up region
        region = np.zeros((n_haps, global_vars.NUM_SNPS, 6), dtype=np.float32)
        # enough SNPs, take middle portion
        if mid >= half_S:
            minor = util.major_minor(
                X[mid - half_S:mid + other_half_S, :, :].transpose(1, 0, 2),
                neg1,
            )
            region[:, :, :] = minor

        # not enough SNPs, need to center-pad
        else:
            print("NOT ENOUGH SNPS", n_sites)
            # print(num_SNPs, S, mid, half_S)
            minor = util.major_minor(X.transpose(1, 0, 2), neg1)
            region[:,
                   half_S - mid:half_S - mid + global_vars.NUM_SNPS, :] = minor
    return region

# def prep_region(ts, neg1, region_len):
#     """Gets simulated data ready"""
#     gt_matrix = ts.genotype_matrix().astype(float)
#     snps_total = gt_matrix.shape[0]

#     positions = [round(variant.site.position) for variant in ts.variants()]
#     assert len(positions) == snps_total
#     dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L for j in
#         range(snps_total-1)]

#     # when mirroring real data
#     return util.process_gt_dist(gt_matrix, dist_vec, region_len=region_len,
#         neg1=neg1)

# testing
if __name__ == "__main__":

    batch_size = 10
    params = param_set.ParamSet()
    print (params)

    # quick test
    print("sim exp")
    generator = Generator(simulation.simulate_exp, ["N1", "kappa"], [20],
                          global_vars.DEFAULT_SEED)
    generator.update_params([params.N1.value, params.kappa.value])
    mini_batch = generator.simulate_batch(batch_size=batch_size)

    print("x", mini_batch.shape)
