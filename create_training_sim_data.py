import param_set
import numpy as np
import math
import msprime
import tqdm
import time
import tskit
import argparse
from real_data_random import get_root_nucleotide_dist
from simulation import simulate_isolated


def create_reference(seq_length: int = 1_000_000):
    nuc_dist = np.array([0.25, 0.25, 0.25, 0.25])
    options = ["A", "C", "G", "T"]
    sequence = np.random.choice(options, p=nuc_dist, size=seq_length)

    return sequence

def main(args):
    params = param_set.ParamSet()

    parameters = ["mu", "rho", "N1", "growth"]#, "conversion", "conversion_length"]
    parameter_values = [1.25e-8, 1.25e-8, 10_000, 1e-2]#, 5e-8, 2]

    params.update(parameters, parameter_values)

    CHROMS = list(map(str, range(1, 23)))
    CHROMS = [f"chr{c}" for c in CHROMS]
    #CHROMS = ["chr1"]
    # simulate a bunch of chromosomes
    for chrom in tqdm.tqdm(CHROMS):
        seed = np.random.randint(0, 2**32)
        # simulate the reference
        reference = create_reference(seq_length=args.length)
        # get the true root distribution on this chromosome
        root_dists = get_root_nucleotide_dist(reference)
        # generate the simulation using the true root dist on this chromosome
        treeseq = simulate_isolated(params, [100], args.length, seed)
        #print (f"{n_sites} variants on {chrom}")
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
