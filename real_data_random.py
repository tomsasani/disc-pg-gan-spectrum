"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
from collections import defaultdict
import numpy as np
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
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

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


def prep_real_region(
    haplotypes: np.ndarray,
    positions: np.ndarray,
    # reference_alleles: np.ndarray,
    # alternate_alleles: np.ndarray,
    # ancestor: mutyper.Ancestor,
    # chrom: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    prepare the feature array for a real region.

    Args:
        haplotypes (np.ndarray): _description_
        positions (np.ndarray): _description_
        n_haps (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

    n_snps, n_haps = haplotypes.shape
    assert n_snps == global_vars.NUM_SNPS

    # X = np.zeros((n_snps, n_haps, 6), dtype=np.float32)

    # loop over SNPs
    # for vi in range(n_snps):
        # only include mutations that occur at a confidently polarized
        # nucleotide in the ancestral reference genome.
        # NOTE: we do this so that we only include mutations that occurred
        # in the space of nucleotides described by the root distribution in
        # the ancestral reference genome sequence.
        # NOTE: always assuming that we're only dealing with biallelics
        # mutation = ancestor.mutation_type(
        #     chrom,
        #     int(positions[vi]), # NOTE: why is explicit int conversion necessary here? it is...
        #     reference_alleles[vi].decode("utf-8"),
        #     alternate_alleles[vi][0].decode("utf-8"),
        # )
        # mut_i = global_vars.MUT2IDX[">".join(mutation)]
        # X[vi, :] = haplotypes[vi]

    X = np.expand_dims(haplotypes, axis=2)

    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    seg = util.find_segregating_idxs(X)

    X_filtered = X[seg, :, :]
    filtered_positions = positions[seg]

    return X_filtered, filtered_positions

def get_root_nucleotide_dist(sequence: str):
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
    counts = Counter(sequence)

    root_dist = np.zeros(4)
    for nuc_i, nuc in enumerate(global_vars.NUC_ORDER):
        root_dist[nuc_i] = counts[nuc]
    # NOTE: maybe don't do this in the future
    if np.sum(root_dist) == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    else: return root_dist / np.sum(root_dist)


class RealDataRandomIterator:

    def __init__(self, hdf_fh: str, ref_fh: str, bed_file: str, seed: int):

        self.rng = np.random.default_rng(seed)

        callset = h5py.File(hdf_fh, mode='r')
        # vcf_ = VCF(vcf_fh, gts012=True)
        # self.vcf = vcf_

        # array of chromosomes
        self.chromosomes = callset['variants/CHROM']

        # array of haplotypes
        raw_gts = callset['calldata/GT']
        newshape = (raw_gts.shape[0], -1)
        self.haplotypes = np.reshape(raw_gts, newshape)
        self.haplotypes[self.haplotypes < 0] = 0

        print("raw", raw_gts.shape)
        print ("new", self.haplotypes.shape)

        self.positions = callset['variants/POS']
        # self.reference_alleles = callset['variants/REF']
        # self.alternate_alleles = callset['variants/ALT']

        self.num_haplotypes = self.haplotypes.shape[1]

        # read in ancestral sequence using mutyper
        # ancestor = mutyper.Ancestor(ref_fh, k=1, sequence_always_upper=True)
        # self.ancestor = ancestor

        AUTOSOMES = list(map(str, range(1, 23)))
        AUTOSOMES = [f"chr{c}" for c in AUTOSOMES]
        self.autosomes = AUTOSOMES

        # # map chromosome names to chromosome lengths
        # seq2len = dict(zip(vcf_.seqnames, vcf_.seqlens))
        # self.sequence_lengths = seq2len

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

        if self.exclude_tree is None:
            return False

        else:
            overlaps = self.exclude_tree[chrom].find(start, end)
            for inter in overlaps:
                total_bp_overlap += inter.end - inter.start

            overlap_pct = total_bp_overlap / (end - start)
            if overlap_pct > 0.5: return True
            else: return False


    def sample_real_region(self) -> np.ndarray:
        """Sample a random "real" region of the genome from the provided VCF file,
        and use the variation data in that region to produce an ndarray of shape
        (n_haps, n_sites, 6), where n_haps is the number of haplotypes in the VCF,
        n_sites is the NUM_SNPs we want in both a simulated or real region (defined
        in global_vars), and 6 is the number of 1-mer mutation types.

        Args:
            start_pos (int, optional): Starting position of the desired region. Defaults to None.

        Returns:
            np.ndarray: np.ndarray of shape (n_haps, n_sites, 6).
            np.ndarray: np.ndarray of shape (4,).
        """

        # grab a random position on the genome,
        start_idx = self.rng.integers(0, self.chromosomes.shape[0] - global_vars.NUM_SNPS)

        # figure out the chromosome on which we're starting
        chromosome, start_pos = self.chromosomes[start_idx].decode("utf-8"), self.positions[start_idx]

        # if the chromosome isn't an autosome, try again
        if chromosome not in self.autosomes:
            return self.sample_real_region()

        # set the ending position to be the starting position + the
        # number of SNPs we want to capture
        end_idx = start_idx + global_vars.NUM_SNPS

        # make sure end idx isn't off the edge of the chromosome
        if end_idx >= self.positions.shape[0] - 1:
            return self.sample_real_region()

        # if we end on a different chromosome, try the whole process again
        if self.chromosomes[end_idx].decode("utf-8") != chromosome:
            return self.sample_real_region()

        end_pos = self.positions[end_idx]
        # check if the region overlaps excluded regions by too much
        excessive_overlap = self.excess_overlap(chromosome, start_pos, end_pos)

        # get haplotypes and positions in this region
        haps = self.haplotypes[start_idx:end_idx]
        sites = self.positions[start_idx:end_idx]

        # make sure positions are sorted
        assert np.all(np.diff(sites) >= 0)

        # if we do have an accessible region
        if not excessive_overlap:
            region, positions = prep_real_region(
                haps,
                sites,
                # self.reference_alleles[start_idx:end_idx],
                # self.alternate_alleles[start_idx:end_idx],
                # self.ancestor,
                # chromosome,
            )
            # sequence = str(self.ancestor[chromosome][start_pos:end_pos].seq).upper()
            # root_dist = get_root_nucleotide_dist(sequence)
            return region, positions

        # try again recursively if not in accessible region
        else:
            return self.sample_real_region()

    def sample_real_region_vcf(self) -> np.ndarray:
        """Sample a random "real" region of the genome from the provided VCF file,
        and use the variation data in that region to produce an ndarray of shape
        (n_haps, n_sites, 6), where n_haps is the number of haplotypes in the VCF,
        n_sites is the NUM_SNPs we want in both a simulated or real region (defined
        in global_vars), and 6 is the number of 1-mer mutation types.

        Args:
            start_pos (int, optional): Starting position of the desired region. Defaults to None.

        Returns:
            np.ndarray: np.ndarray of shape (n_haps, n_sites, 6).
            np.ndarray: np.ndarray of shape (4,).
        """

        # first, choose a random chromosome. we'll randomly sample the chromosome
        # names, weighted by their overall lengths. 
        chromosomes, lengths = [], []
        for chrom, length in self.sequence_lengths.items():
            if chrom not in self.autosomes: continue
            chromosomes.append(chrom)
            lengths.append(length)

        lengths_arr = np.array(lengths)
        lengths_probs = lengths_arr / np.sum(lengths_arr)

        chromosome = self.rng.choice(chromosomes, size=1, p=lengths_probs)[0]
        chromosome_len = self.sequence_lengths[chromosome]

        # grab a random position on this chromosome, such that the position
        # plus the desired region length doesn't exceed the length of the chromosome.
        start_pos = self.rng.integers(0, chromosome_len - global_vars.NUM_SNPS)

        region = np.zeros((global_vars.NUM_SNPS, self.num_haplotypes, 6))
        positions = np.zeros(global_vars.NUM_SNPS, dtype=np.int64)
        # keep track of how many variants we've encountered while iterating
        snp_count = 0
        # initialize a region extending from this position to the end of the chromosome
        # we probably won't need such a large region.
        region_str = f"{chromosome}:{start_pos}-{chromosome_len}"
        end_pos = start_pos + 1
        for v in self.vcf(region_str):
            if snp_count >= global_vars.NUM_SNPS: break
            haplotypes = np.array(v.genotypes)[:, :-1].flatten()
            ref, alt = v.REF.upper(), v.ALT[0].upper()
            mutation = self.ancestor.mutation_type(v.CHROM, v.end, ref, alt)
            mut_idx = global_vars.MUT2IDX[">".join(mutation)]
            positions[snp_count] = v.POS
            region[snp_count, :, mut_idx] = haplotypes
            snp_count += 1

        if snp_count < global_vars.NUM_SNPS:
            return self.sample_real_region_vcf()

        # make sure positions are sorted
        assert np.all(np.diff(positions) >= 0)

        # redefine start and end
        start_pos, end_pos = np.min(positions), np.max(positions)
        excessive_overlap = self.excess_overlap(chromosome, start_pos, end_pos)

        # if we do have an accessible region
        if not excessive_overlap:
            sequence = str(self.ancestor[chromosome][start_pos:end_pos].seq).upper()
            root_dist = get_root_nucleotide_dist(sequence)
            return region, root_dist, positions

        # try again recursively if not in accessible region
        else:
            return self.sample_real_region_vcf()



    def real_batch(
        self,
        norm_len: int,
        batch_size=global_vars.BATCH_SIZE,
    ):

        # store the actual haplotype "images" in each batch within this region
        regions = np.zeros(
            (
                batch_size,
                self.num_haplotypes,
                global_vars.NUM_SNPS,
                global_vars.NUM_CHANNELS,
            ),
            dtype=np.float32,
        )

        # store the root distribution of nucleotides in each region
        # root_dists = np.zeros((batch_size, 4))
        # store the lengths of the sampled regions
        region_lens = np.zeros(batch_size)

        for i in range(batch_size):
            region, positions = self.sample_real_region()
            region_lens[i] = np.max(positions) - np.min(positions)
            fixed_region = util.process_region(region, positions, norm_len)
            regions[i] = fixed_region
            # root_dists[i] = root_dist

        return regions, region_lens


if __name__ == "__main__":
    # test file
    hdf_fh = sys.argv[1]
    ref_fh = sys.argv[2]
    bed_fh = sys.argv[3]
    iterator = RealDataRandomIterator(hdf_fh, ref_fh, bed_fh, global_vars.DEFAULT_SEED)

    start_time = datetime.datetime.now()
    for i in range(100):
        region = iterator.sample_real_region(False)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print("time s:ms", elapsed.seconds,":",elapsed.microseconds)
