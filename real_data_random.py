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
    reference_alleles: np.ndarray,
    alternate_alleles: np.ndarray,
    ancestor: mutyper.Ancestor,
    chrom: str,
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

    X = np.zeros((n_snps, n_haps, 6), dtype=np.float32)

    # loop over SNPs
    for vi in range(n_snps):
        # only include mutations that occur at a confidently polarized
        # nucleotide in the ancestral reference genome.
        # NOTE: we do this so that we only include mutations that occurred
        # in the space of nucleotides described by the root distribution in
        # the ancestral reference genome sequence.
        # NOTE: always assuming that we're only dealing with biallelics
        mutation = ancestor.mutation_type(
            chrom,
            int(positions[vi]), # NOTE: why is explicit int conversion necessary here? it is...
            reference_alleles[vi].decode("utf-8"),
            alternate_alleles[vi][0].decode("utf-8"),
        )
        mut_i = global_vars.MUT2IDX[">".join(mutation)]
        X[vi, :, mut_i] = haplotypes[vi]

    seg = util.find_segregating_idxs(X)
    # if seg.shape[0] < 36:
    #     print (f"{seg.shape[0] / 36} segregating sites in real data")

    X_filtered = haplotypes[seg, :]
    filtered_positions = positions[seg]

    return haplotypes, positions

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

    def __init__(self, hdf_fh: str, bed_file: str, seed: int, use_full_spectrum: bool = False):

        self.rng = np.random.default_rng(seed)

        callset = h5py.File(hdf_fh, mode='r')

        # array of chromosomes
        self.chromosomes = callset['variants/CHROM']
        self.use_full_spectrum = use_full_spectrum

        # array of haplotypes
        raw_gts = callset['calldata/GT']

        newshape = (raw_gts.shape[0], -1)
        # NOTE: processing so that we have individual haplotypes
        haplotypes = np.reshape(raw_gts, newshape)
        # grab half of the haplotypes to use for the iterator.
        # NOTE: this is only necessary if we use the simulated data,
        # since the simulated data only report a single allele at each
        # site in the VCF. if we're using "real" data, we can simply reshape
        # the numpy array.
        self.haplotypes = haplotypes
        print (np.any(haplotypes < 0))
        #self.haplotypes[self.haplotypes < 0] = 0

        print("raw", raw_gts.shape)
        print ("new", self.haplotypes.shape)

        self.positions = callset['variants/POS']

        self.num_haplotypes = self.haplotypes.shape[1]

        # read in ancestral sequence using mutyper
        ancestor = mutyper.Ancestor(ref_fh, k=1, sequence_always_upper=True)
        self.ancestor = ancestor

        AUTOSOMES = list(map(str, range(1, 23)))
        AUTOSOMES = [f"chr{c}" for c in AUTOSOMES]
        self.autosomes = AUTOSOMES

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
                self.reference_alleles[start_idx:end_idx],
                self.alternate_alleles[start_idx:end_idx],
                self.ancestor,
                chromosome,
            )
            sequence = str(self.ancestor[chromosome][start_pos:end_pos].seq).upper()
            root_dist = get_root_nucleotide_dist(sequence)
            return region, root_dist, positions

        # try again recursively if not in accessible region
        else:
            return self.sample_real_region()


    def real_batch(
        self,
        norm_len: int,
        batch_size=global_vars.BATCH_SIZE,
    ):

        num_channels = 7 if self.use_full_spectrum else 2
        # store the actual haplotype "images" in each batch within this region
        regions = np.zeros(
            (
                batch_size,
                self.num_haplotypes,
                global_vars.NUM_SNPS,
                num_channels,
            ),
            dtype=np.float32,
        )

        # store the root distribution of nucleotides in each region
        root_dists = np.zeros((batch_size, 4))
        # store the lengths of the sampled regions
        region_lens = np.zeros(batch_size)

        for i in range(batch_size):
            region, root_dist, positions = self.sample_real_region()
            region_lens[i] = positions[-1] - positions[0]
            fixed_region = util.process_region(region, positions, norm_len, use_full_spectrum = self.use_full_spectrum,)
            regions[i] = fixed_region
            root_dists[i] = root_dist

        return regions, root_dists, region_lens


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
