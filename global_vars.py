'''For collecting global values'''
# section A: general -----------------------------------------------------------
#NUM_SNPS = 36 * 1      # number of seg sites, should be divisible by 4
BATCH_SIZE = 20 # number of real/simulated regions to sample in a given batch

WINDOW_SIZE = 4
NUM_WINDOWS = 36
NUM_SNPS = 36
NUM_NODE_FEATURES = 1 # features stored in each node (currently, just one nucleotide per node)

DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

MUT2IDX = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
REVCOMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
NUC_ORDER = ["A", "C", "G", "T"]
NUC2IDX = dict(zip(NUC_ORDER, range(4)))

NUM_CHANNELS = 7

# section B: overwriting in-file data-------------------------------------------

# to use custom trial data, switch OVERWRITE_TRIAL_DATA to True and
# change the TRIAL_DATA dictionary to have the values desired.
# Model, params, and param_values must be defined
OVERWRITE_TRIAL_DATA = False
TRIAL_DATA = { 'model': 'const', 'params': 'N1', 'data_h5': None,
               'bed_file': None, 'reco_folder': None, 'param_values': '1000'}

'''The high-coverage data ("new data") appears to have partial filtering on
singletons. It is recommended, if using the high-coverage data, to enable
singleton filtering for both real and simulated data. It may be necessary to
experiment with different filtering rates.'''
FILTER_SIMULATED = False
FILTER_REAL_DATA = False
FILTER_RATE = 0.50
NUM_SNPS_ADJ = NUM_SNPS * 3
# ------------------------------------------------------------------------------

def get_reco_files(reco_folder):
    # DEFAULT IS FOR hg19 FORMAT
    files = [reco_folder + "genetic_map_GRCh37_chr" + str(i) +
             ".txt" for i in HUMAN_CHROM_RANGE]

    # for high coverage/ hg38, comment the above line, and uncomment the following:
    # pop = reco_folder[-4: -1]
    # files = [reco_folder + pop + "_recombination_map_hapmap_format_hg38_chr_" + str(i) +
    #          ".txt" for i in HUMAN_CHROM_RANGE]

    return files


EDGE_IDXS = []
# enumerate all possible edge connections
for edge_idx in range(NUM_SNPS - 1):
    # ref at idx 0 can connect to ref at idx 2 or alt at idx 3 (i.e., next site)
    for base_idx in (edge_idx * 2, edge_idx * 2 + 1):
        if base_idx % 2 == 0:
            EDGE_IDXS.append([base_idx, base_idx + 2])
            EDGE_IDXS.append([base_idx, base_idx + 3])
        else:
            EDGE_IDXS.append([base_idx, base_idx + 1])
            EDGE_IDXS.append([base_idx, base_idx + 2])