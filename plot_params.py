import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc("font", size=12)

df = pd.read_csv("summary.csv")

n_params = len(df["param"].unique())

subplots = n_params + 2

f, axarr = plt.subplots(subplots, figsize=(15, 15))

exp_param_vals = {
    "N1": 9_000,
    "N2": 5_000,
    "T1": 2_000,
    "T2": 350,
    "mu": 1.25e-8,
}

exp_param_ranges = {
    "N1": (1_000, 30_000),
    "N2": (1_000, 30_000),
    "T1": (1_500, 5_000),
    "T2": (100, 1_500),
    "mu": (1e-9, 1e-7),
}


loss_df = df.drop_duplicates("epoch")
loss_df = loss_df.replace(to_replace={-1: np.nan})
loss_df.ffill(inplace=True)

print (loss_df)
axarr[0].plot(loss_df["epoch"], loss_df["generator_loss"])
axarr[0].set_ylabel("Generator loss")
axarr[1].plot(loss_df["epoch"], loss_df["Real acc"], label="Real data")
axarr[1].plot(loss_df["epoch"], loss_df["Fake acc"], label="Fake data")
axarr[1].set_ylabel("Discriminator accuracy")
axarr[1].axhline(0.5, c="k", ls=":")
axarr[1].legend()

for idx in (0, 1):
    sns.despine(ax=axarr[idx], top=True, right=True)


cur_idx = 2
for param in df["param"].unique():
    sub_df = df[df["param"] == param]
    axarr[cur_idx].plot(sub_df["epoch"], sub_df["param_value"], c="dodgerblue", label="inferred", lw=2)
    axarr[cur_idx].axhline(exp_param_vals[param], c="firebrick", ls="--", label="expected")
    lower, upper = exp_param_ranges[param]
    axarr[cur_idx].set_ylim(lower, upper)
    axarr[cur_idx].set_ylabel(param)
    sns.despine(ax=axarr[cur_idx], top=True, right=True)
    axarr[cur_idx].legend()
    cur_idx += 1
axarr[-1].set_xlabel("Epoch")
f.tight_layout()
f.savefig('o.png', dpi=200)