'''For collecting global values'''
# section A: general -----------------------------------------------------------
#NUM_WINDOWS = 36
#WINDOW_SIZE = 10
NUM_SNPS = 36 * 4       # number of seg sites, should be divisible by 4
L = 50_000           # heuristic to get enough SNPs for simulations (50,000 or fifty-thousand)
BATCH_SIZE = 20 # number of real/simulated regions to sample in a given batch

DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

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