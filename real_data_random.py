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
import mutyper
from collections import Counter

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
            interval = Interval(int(start), int(end))
            tree[chrom].insert_interval(interval)

    return tree

def prep_real_region(vcf: VCF, ancestor: mutyper.Ancestor, chrom: str, start: int, end: int, n_haps: int) -> np.ndarray:
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
        # only include mutations that occur at a confidently polarized
        # nucleotide in the ancestral reference genome.
        # NOTE: we do this so that we only include mutations that occurred
        # in the space of nucleotides described by the root distribution in
        # the ancestral reference genome sequence.
        mutation = ancestor.mutation_type(v.CHROM, v.end, ref, alt)
        if None in mutation: continue
        mut_i = mut2idx[">".join(mutation)]

        vi = v.start - start
        out_arr[vi, :, mut_i] = np.array(v.genotypes)[:, :-1].flatten()
        called_positions[vi] = 1

    # subset out_arr to the places (out of size end - start) where
    # we actually called a mutation
    out_arr_sub = out_arr[np.where(called_positions)[0], :, :]
    return out_arr_sub

def get_root_nucleotide_dist(ancestor: mutyper.Ancestor, chrom: str, start: int, end: int) -> np.ndarray:
    """
    Given an interval, figure out the frequency of each nucleotide within the
    region using an ancestral reference genome sequence. This is important so that we
    can parametrize the Generator such that mutations are simulated w/r/t an identical
    starting nucleotide distribution.

    Args:
        chrom (str): Chromosome of query region.
        start (int): Starting position of query region.
        end (int): Ending position of query region.

    Returns:
        np.ndarray: 1-D numpy array containing frequencies of A,T,C and G nucs.
    """
    nuc_order = ["A", "C", "G", "T"]
    sequence = str(ancestor[chrom][start:end].seq).upper()
    counts = Counter(sequence)

    root_dist = np.zeros(4)
    for nuc_i, nuc in enumerate(nuc_order):
        root_dist[nuc_i] = counts[nuc]
    # NOTE: maybe don't do this in the future
    if np.sum(root_dist) == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    else: return root_dist / np.sum(root_dist)


class RealDataRandomIterator:

    def __init__(self, vcf_fh: str, ref_fh: str, bed_file: str, seed: int):
        self.rng = default_rng(seed)

        vcf = VCF(vcf_fh)
        self.vcf_ = vcf
        self.num_haplotypes = len(vcf.samples) * 2

        # read in ancestral sequence using mutyper
        ancestor = mutyper.Ancestor(ref_fh, k=1, sequence_always_upper=True)
        self.ancestor = ancestor

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
        start_pos: int = None,
        autosomes_only: bool = True,
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
            np.ndarray: np.ndarray of shape (4,).
        """

        # first, choose a random chromosome. we'll randomly sample the chromosome
        # names, weighted by their overall lengths.  
        CHROMS = list(map(str, range(1, 23)))
        CHROMS = [f"chr{c}" for c in CHROMS]

        chromosomes, lengths = [], []
        for chrom, length in self.sequence_lengths.items():
            if autosomes_only and chrom not in CHROMS: continue
            chromosomes.append(chrom)
            lengths.append(length)

        lengths_arr = np.array(lengths)
        lengths_probs = lengths_arr / np.sum(lengths_arr)

        chromosome = self.rng.choice(chromosomes, size=1, p=lengths_probs)[0]
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
            region = prep_real_region(
                self.vcf_,
                self.ancestor,
                chromosome,
                start_pos,
                end_pos,
                self.num_haplotypes,
            )
            fixed_region = util.process_region(region, neg1=neg1)
            root_dists = get_root_nucleotide_dist(self.ancestor, chromosome, start_pos, end_pos)
            return fixed_region, root_dists

        # try again recursively if not in accessible region
        else:
            return self.sample_real_region(neg1, region_len)

    def real_batch(
        self,
        region_len: int,
        batch_size=global_vars.BATCH_SIZE,
        neg1=True,
    ):

        # store the actual haplotype "images" in each batch within this region
        regions = np.zeros(
            (batch_size, self.num_haplotypes, global_vars.NUM_SNPS, 6),
            dtype=np.float32,
        )

        # store the root distribution of nucleotides in each region
        root_dists = np.zeros((batch_size, 4))

        for i in range(batch_size):
            region_img, region_nucs = self.sample_real_region(neg1, region_len)
            regions[i] = region_img
            root_dists[i] = region_nucs

        return regions, root_dists


if __name__ == "__main__":
    # test file
    vcf_fh = sys.argv[1]
    ref_fh = sys.argv[2]
    bed_fh = sys.argv[3]
    iterator = RealDataRandomIterator(vcf_fh, ref_fh, bed_fh, global_vars.DEFAULT_SEED)

    start_time = datetime.datetime.now()
    for i in range(100):
        region = iterator.sample_real_region(False, 50_000)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print("time s:ms", elapsed.seconds,":",elapsed.microseconds)
