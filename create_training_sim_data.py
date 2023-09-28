import param_set
import numpy as np
import math
import msprime
import tqdm
import time
import tskit
import argparse
from real_data_random import get_root_nucleotide_dist
from simulation import simulate_exp


def create_reference(seq_length: int = 1_000_000):
    nuc_dist = np.array([0.2, 0.3, 0.3, 0.2])
    options = ["A", "C", "G", "T"]
    sequence = np.random.choice(options, p=nuc_dist, size=seq_length)

    return sequence

def main(args):
    params = param_set.ParamSet()

    parameters = ["mu", "rho", "N1", "N2", "T1", "T2", "growth"]#, "conversion", "conversion_length"]
    parameter_values = [1e-8, 1e-8, 9_000, 5_000, 2_000, 350, 5e-3]#, 5e-8, 2]

    params.update(parameters, parameter_values)

    print (f"Creating VCF and FA for {args.chrom} with length {args.length}")

    # simulate a chromosome
    seed = np.random.randint(0, 2**32)
    # simulate the reference
    reference = create_reference(seq_length=args.length)
    # get the true root distribution on this chromosome
    root_dists = get_root_nucleotide_dist(reference)
    # generate the simulation using the true root dist on this chromosome
    treeseq = simulate_exp(
        params,
        [100],
        root_dists,
        args.length,
        seed,
        adj_mut=args.adj_mut,
        adj_rate=args.adj_rate,
        restrict_time=args.restrict_time,
    )
    #print (f"{n_sites} variants on {chrom}")
    # first convert to VCF
    with open(f"data/simulated/vcf/{args.outpref}.vcf", "w") as outfh:
        treeseq.write_vcf(outfh, contig_id=args.chrom)
    # update reference sequence

    site_table = treeseq.tables.sites
    positions = site_table.position.astype(np.int64)
    reference_alleles = tskit.unpack_strings(site_table.ancestral_state, site_table.ancestral_state_offset)

    # refactor reference using true ancestral alleles and create hdf5
    reference[positions] = reference_alleles
    with open(f"data/simulated/ref/{args.outpref}.fa", "w") as outfh:
        reference_seq = "".join(reference)
        outfh.write(f">{args.chrom}\n{reference_seq}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chrom")
    p.add_argument("-length", default=1_000_000, type=int)
    p.add_argument("-adj_mut", type=float, default=1.)
    p.add_argument("-adj_rate", type=float, default=1.)
    p.add_argument("-restrict_time", type=bool, default=False)
    p.add_argument("-outpref", type=str)

    args = p.parse_args()
    main(args)
