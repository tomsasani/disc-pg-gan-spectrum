import param_set
from generator import Generator
import numpy as np
import math
import msprime
import h5py
import allel
import tqdm 
import time 
import tskit 
import numba 
import argparse

def create_reference(root_distribution: np.ndarray, seq_length: int = 1_000_000):
    
    options = ["A", "C", "G", "T"]

    sequence = np.random.choice(options, p=root_distribution, size=seq_length)
    
    return sequence

def simulate_exp(params, sample_sizes, root_distribution, seed, seq_length: int = 1_000_000):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N0, growth_rate=params.growth.value,)
    demography.add_population_parameters_change(population="A", time=T2, initial_size=N2, growth_rate=0,)
    demography.add_population_parameters_change(population="A", time=params.T1.value, initial_size=params.N1.value,)


    ts = msprime.sim_ancestry(
        samples=sum(sample_sizes),
        demography=demography,
        sequence_length=seq_length,
        recombination_rate=params.rho.value,
        gene_conversion_rate=params.conversion.value,
        gene_conversion_tract_length=params.conversion_length.value,
        discrete_genome=True,
        random_seed=seed,
    )

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=msprime.HKY(root_distribution=root_distribution, kappa=2.),
        random_seed=seed,
        discrete_genome=True,
    )

    return mts

def main(args):
    params = param_set.ParamSet()

    nucs = ["A", "C", "G", "T"]
    options = np.arange(1, 5)
    option2nuc = dict(zip(options, nucs))

    parameters = ["mu", "rho", "T1", "T2", "N1", "N2"]#, "conversion", "conversion_length"]
    parameter_values = [5e-9, 5e-9, 2_000, 350, 9_000, 5_000]#, 5e-8, 2]

    params.update(parameters, parameter_values)

    root_dists = np.array([0.25, 0.25, 0.25, 0.25])

    CHROMS = list(map(str, range(1, 23)))
    CHROMS = [f"chr{c}" for c in CHROMS]
    # simulate a bunch of chromosomes
    for chrom in tqdm.tqdm(CHROMS):

        cur_time = time.time()
        # generate the simulation
        treeseq = simulate_exp(params, [100], root_dists, 4242, seq_length=args.length)
        site_table = treeseq.tables.sites
        positions = site_table.position.astype(np.int64)
        reference_alleles = tskit.unpack_strings(site_table.ancestral_state, site_table.ancestral_state_offset)
        # generate a toy reference genome using the specified root distribution
        reference = create_reference(root_dists, seq_length=args.length)
        #reference = np.array([option2nuc[o] for o in reference])
        # refactor reference using true ancestral alleles and create hdf5

        # first convert to VCF
        with open(f"data/simulated/vcf/{chrom}.simulated.vcf", "w") as outfh:
            treeseq.write_vcf(outfh, contig_id=chrom)
        # update reference sequence

        reference[positions] = reference_alleles
        with open(f"data/simulated/ref/{chrom}.simulated.fa", "w") as outfh:
            reference_seq = "".join(reference)
            outfh.write(f">{chrom}\n{reference_seq}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-length", default=1_000_000, type=int)
    args = p.parse_args()
    main(args)

# # convert to h5
# allel.vcf_to_hdf5(
#     "data/vcf/simulated.vcf",
#     "data/vcf/simulated.h5",
#     fields=['CHROM', 'GT', 'POS', 'REF', 'ALT'],
#     overwrite=True,
# )
