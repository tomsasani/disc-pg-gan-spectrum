"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
from collections import defaultdict
import numpy as np
from numpy.random import default_rng
import sys
import datetime
from bx.intervals.intersection import Interval, IntervalTree
import gzip
import csv
from collections import defaultdict
from cyvcf2 import VCF
import tqdm

# our imports
import global_vars
import util

class Region:

    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = str(chrom)
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.region_len = end_pos - start_pos # L


def read_exclude(fh: str) -> IntervalTree:
    """
    Read in a BED file containing genomic regions from which we want
    to exclude potential variants. Riley et al. 2023 use a callability mask,
    but for my purposes I'll stick to a known file (Heng Li's LCR file for 
    hg19/hg38).

    Args:
        fh (str): Path to filename containing regions. Must be BED-formatted. Can be \
            uncompressed or gzipped.

    Returns:
        tree (Dict[IntervalTree]): Dictionary of IntervalTree objects, with one key per \
            chromosome. Each IntervalTree containing the BED regions from `fh`, on which we can \
            quickly perform binary searches later.
    """

    tree = defaultdict(IntervalTree)
    is_zipped = fh.endswith(".gz")

    print ("BUILDING EXCLUDE TREE")

    with gzip.open(fh, "rt") if is_zipped else open(fh, "rt") as infh:
        csvf = csv.reader(infh, delimiter="\t")
        for l in tqdm.tqdm(csvf):
            if l[0].startswith("#") or l[0] == "chrom": continue
            chrom, start, end = l
            if chrom != "1": continue
            interval = Interval(int(start), int(end))
            tree[chrom].insert_interval(interval)

    return tree

def prep_real_region(vcf: VCF, chrom: str, start: int, end: int, n_haps: int) -> np.ndarray:
    revcomp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    mut2idx = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))

    out_arr = np.zeros((end - start, n_haps, 6), dtype=np.float32)
    called_positions = np.zeros(end - start, dtype=np.int8)

    region_str = f"{chrom}:{start}-{end}"

    # TODO: write function to filter variants (FILTER, etc.) on the fly,
    # rather than requiring preprocessing using e.g., bcftools prior.
    for v in vcf(region_str):
        # require single-nucleotide variants
        if v.var_type != "snp": continue
        # filter out multi-allelics
        if len(v.ALT) > 1: continue
        ref, alt = v.REF.upper(), v.ALT[0].upper()
        if ref in ("G", "T"):
            ref, alt = revcomp[ref], revcomp[alt]
        mutation = ">".join([ref, alt])
        mut_i = mut2idx[mutation]

        vi = v.start - start
        out_arr[vi, :, mut_i] = np.array(v.genotypes)[:, :-1].flatten()
        called_positions[vi] = 1

    # subset out_arr to the places (out of size end - start) where
    # we actually called a mutation
    out_arr_sub = out_arr[np.where(called_positions)[0], :, :]
    return out_arr_sub


class RealDataRandomIterator:

    def __init__(self, vcf_fh: str, bed_file: str, seed: int):
        self.rng = default_rng(seed)

        vcf = VCF(vcf_fh)
        self.vcf_ = vcf
        self.num_haplotypes = len(vcf.samples) * 2

        # map chromosome names to chromosome lengths
        seq2len = dict(zip(vcf.seqnames, vcf.seqlens))
        self.sequence_lengths = seq2len

        # exclude regions
        self.exclude_tree = read_exclude(bed_file) if bed_file is not None else None

    def excess_overlap(self, chrom: str, start: int, end: int) -> bool:
        """
        Given an interval, figure out how much it overlaps an exclude region
        (if at all). If it overlaps the exclude region by more than 50% of its
        length, ditch the interval.

        Args:
            chrom (str): Chromosome of query region.
            start (int): Starting position of query region.
            end (int): Ending position of query region.
            thresh (float, optional): Fraction of base pairs of overlap between query region \
                and exclude regions, at which point we should ditch the interval. Defaults to 0.5.

        Returns:
            overlap_is_excessive (bool): Whether the query region overlaps the exclude \
                regions by >= 50%.
        """
        total_bp_overlap = 0

        overlaps = self.exclude_tree[chrom].find(start, end)
        for inter in overlaps:
            total_bp_overlap += inter.end - inter.start

        overlap_pct = total_bp_overlap / (end - start)
        if overlap_pct > 0.5: return True
        else: return False

    def sample_real_region(
        self,
        neg1: bool,
        region_len: int,
        start_pos: int =None,
    ) -> np.ndarray:
        """Sample a random "real" region of the genome from the provided VCF file,
        and use the variation data in that region to produce an ndarray of shape
        (n_haps, n_sites, 6), where n_haps is the number of haplotypes in the VCF,
        n_sites is the NUM_SNPs we want in both a simulated or real region (defined
        in global_vars), and 6 is the number of 1-mer mutation types.

        Args:
            neg1 (bool): Whether to store minor alleles as -1 (and major alleles as +1).
            region_len (int): Desired length of the region (in base pairs) to query.
            start_pos (int, optional): Starting position of the desired region. Defaults to None.

        Returns:
            np.ndarray: np.ndarray of shape (n_haps, n_sites, 6).
        """

        # first, choose a random chromosome. we'll randomly sample the chromosome
        # names, weighted by their overall lengths.
        chromosomes, lengths = (
            list(self.sequence_lengths.keys()),
            list(self.sequence_lengths.values()),
        )

        lengths_arr = np.array(lengths)
        lengths_probs = lengths_arr / np.sum(lengths_arr)

        chromosome = self.rng.choice(chromosomes, size=1, p=lengths_probs)[0]
        chromosome = "1"
        chromosome_len = self.sequence_lengths[chromosome]

        if start_pos is None:
            # grab a random position on this chromosome, such that the position
            # plus the desired region length doesn't exceed the length of the chromosome.
            start_pos = self.rng.integers(0, chromosome_len - region_len)

        end_pos = start_pos + region_len
        # initialize a Region object for this interval
        region = Region(chromosome, start_pos, end_pos)
        # check if the region overlaps excluded regions by too much
        excessive_overlap = self.excess_overlap(chromosome, start_pos, end_pos)
        # if we do have an accessible region
        if not excessive_overlap:
            region = prep_real_region(self.vcf_, chromosome, start_pos, end_pos, self.num_haplotypes)
            fixed_region = util.process_region(region, neg1=neg1)
            return fixed_region

        # try again recursively if not in accessible region
        else:
            return self.sample_real_region(neg1, region_len)

    def real_batch(
        self,
        region_len: int,
        batch_size=global_vars.BATCH_SIZE,
        neg1=True,
    ):

        regions = np.zeros(
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, 6),
            dtype=np.float32,
        )

        for i in range(batch_size):
            regions[i] = self.sample_real_region(neg1, region_len)

        return regions


if __name__ == "__main__":
    # test file
    filename = sys.argv[1]
    bed_file = sys.argv[2]
    iterator = RealDataRandomIterator(filename, global_vars.DEFAULT_SEED, bed_file)

    start_time = datetime.datetime.now()
    for i in range(100):
        region = iterator.sample_real_region(False, 50_000)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print("time s:ms", elapsed.seconds,":",elapsed.microseconds)
