import param_set
import numpy as np
import math
import msprime
import tqdm
import time
import tskit
import argparse
from simulation import parameterize_mutation_model
from real_data_random import get_root_nucleotide_dist


def simulate_exp(
    params,
    sample_sizes,
    root_distribution,
    seed,
    seq_length: int = 1_000_000,
):
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
        discrete_genome=True,
        random_seed=seed,
    )

    # define mutation model
    mutation_model = parameterize_mutation_model(root_distribution)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=True,
    )

    return mts

def create_reference(seq_length: int = 1_000_000):
    nuc_dist = np.array([0.25, 0.25, 0.25, 0.25])
    options = ["A", "C", "G", "T"]
    sequence = np.random.choice(options, p=nuc_dist, size=seq_length)

    return sequence

def main(args):
    params = param_set.ParamSet()

    parameters = ["mu", "rho", "T1", "T2", "N1", "N2"]#, "conversion", "conversion_length"]
    parameter_values = [5e-9, 5e-9, 2_000, 350, 9_000, 5_000]#, 5e-8, 2]

    params.update(parameters, parameter_values)

    CHROMS = list(map(str, range(1, 23)))
    CHROMS = [f"chr{c}" for c in CHROMS]
    # simulate a bunch of chromosomes
    for chrom in tqdm.tqdm(CHROMS):
        # simulate the reference
        reference = create_reference(seq_length=args.length)
        # get the true root distribution on this chromosome
        root_dists = get_root_nucleotide_dist(reference)
        # generate the simulation using the true root dist on this chromosome
        treeseq = simulate_exp(params, [100], root_dists, 4242, seq_length=args.length)

        # first convert to VCF
        with open(f"data/simulated/vcf/{chrom}.simulated.vcf", "w") as outfh:
            treeseq.write_vcf(outfh, contig_id=chrom)
        # update reference sequence

        site_table = treeseq.tables.sites
        positions = site_table.position.astype(np.int64)
        reference_alleles = tskit.unpack_strings(site_table.ancestral_state, site_table.ancestral_state_offset)
        # refactor reference using true ancestral alleles and create hdf5
        reference[positions] = reference_alleles
        with open(f"data/simulated/ref/{chrom}.simulated.fa", "w") as outfh:
            reference_seq = "".join(reference)
            outfh.write(f">{chrom}\n{reference_seq}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-length", default=1_000_000, type=int)
    args = p.parse_args()
    main(args)
