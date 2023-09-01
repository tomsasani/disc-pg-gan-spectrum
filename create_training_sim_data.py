import param_set
from generator import Generator
import numpy as np
import math
import msprime
import h5py
import allel

def create_reference(root_distribution: np.ndarray, seq_length: int = 1_000_000):
    nucleotides = ["A", "C", "G", "T"]

    reference = []
    for i in range(seq_length):
        nucleotide = np.random.choice(nucleotides, p=root_distribution)[0]
        reference.append(nucleotide)
    return reference

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

batch_size = 10
params = param_set.ParamSet()

LENGTH = 1_000_000

parameters = ["mu", "rho", "T1", "T2", "N1", "N2", "conversion", "conversion_length"]
parameter_values = [1.5e-8, 2e-8, 2_000, 350, 9_000, 5_000, 5e-8, 2]

params.update(parameters, parameter_values)

root_dists = np.array([0.25, 0.25, 0.25, 0.25])

# generate the simulation
treeseq = simulate_exp(params, [100], root_dists, 4242, seq_length=LENGTH)

# generate a toy reference genome using the specified root distribution
reference = create_reference(root_dists, seq_length=LENGTH)

# refactor reference using true ancestral alleles and create hdf5

# first convert to VCF
with open("data/vcf/simulated.vcf", "w") as outfh:
    treeseq.write_vcf(outfh, contig_id="chr1")

# update reference sequence
for var in treeseq.variants():
    ref = var.alleles[0]
    pos = int(var.site.position)
    # update reference at position
    reference[pos] = ref

with open("data/ref/simulated.fa", "w") as outfh:
    reference_seq = "".join(reference)
    outfh.write(f">chr1\n{reference_seq}")

# conver to h5
allel.vcf_to_hdf5(
    "data/vcf/simulated.vcf",
    "data/vcf/simulated.h5",
    fields=['CHROM', 'GT', 'POS', 'REF', 'ALT'],
    overwrite=True,
)
