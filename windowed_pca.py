import h5py
import numpy as np
import mutyper
from collections import defaultdict, Counter
from bx.intervals.intersection import Interval, IntervalTree
import gzip
import csv
import tqdm
from real_data_random import RealDataRandomIterator
import simulation
import generator
import param_set
import pca_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

MUT2IDX = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
NUC_ORDER = ["A", "T", "C", "G"]

hdf_fh = "data/simulated/simulated.mutator.h5"
ref_fh = "data/simulated/simulated.mutator.fa"

params = ["N1", "N2", "T1", "T2"]
param_current = [9_000, 5_000, 2_000, 350]

# NOTE: to what degree does the quality of parameter inference
# affect these estimates? i.e., if we do a bad job of inferring
# the parameters? let's add some random noise to these to simulate
# bad inference

# define iterator
iterator = RealDataRandomIterator(hdf_fh, ref_fh, None, 1234)
n_sites = iterator.positions.shape[0]

simulator = simulation.simulate_exp
sample_sizes = [iterator.num_haplotypes]

print (f"GENERATOR is simulating {sample_sizes[0]} haplotypes")

# set up parameters
sim_params = param_set.ParamSet()
#sim_params.update(params, param_current)

# loop over windows
WINDOW_SIZE = 100

windows = np.arange(0, n_sites, step=WINDOW_SIZE)
window_starts = windows[:-1]

out_df = []

N_SIM = 20

for i, ws in tqdm.tqdm(enumerate(window_starts)):
    region, root_dist, positions = iterator.sample_real_region(ws, num_snps = WINDOW_SIZE, check_chroms = False)
    region_len = np.max(positions) - np.min(positions)
    real_spectrum = np.sum(region, axis=0)

    for sim_iter in range(N_SIM):
        # simulate tree sequence from
        seed = np.random.randint(1, 2**32)
        ts = simulator(
            sim_params,
            [iterator.num_haplotypes // 2],
            root_dist,
            region_len,
            seed,
        )

        # get array
        simulated_region, simulated_positions = generator.prep_simulated_region(ts)
        sim_spectrum = np.sum(simulated_region, axis=0)

        a_rescaled, b_rescaled = pca_utils.rescale_window_pair(real_spectrum, sim_spectrum)
        distance = pca_utils.compare_windows(a_rescaled, b_rescaled)
        out_df.append({
            "window_start": i,
            #"window_end": positions[-1],
            "distance": distance,
            "iter": sim_iter
        })

out_df = pd.DataFrame(out_df)

x, y, yerr = [], [], []

out_df_grouped = out_df.groupby("window_start").agg({"distance": [np.mean, ss.sem, np.max, np.median, np.std]}).reset_index()
out_df_grouped.columns = ["start", "mean", "sem", "max", "median", "std"]


f, (ax1, ax2) = plt.subplots(2, figsize=(14, 6))
ax1.errorbar(out_df_grouped["start"], out_df_grouped["median"], yerr=out_df_grouped["std"], fmt="o", lw=2, capsize=6, capthick=2)
ax2.scatter(out_df_grouped["start"], out_df_grouped["max"])

f.tight_layout()
f.savefig("o.png", dpi=200)

