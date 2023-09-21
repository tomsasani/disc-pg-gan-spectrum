import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_snps, n_haps, n_nucs = 36, 200, 4

# assume we've summed across haplotypes
X = np.zeros((4, n_snps))

for snp_i in np.arange(n_snps):
    # pick two random nucleotides
    nuc_i = np.random.choice(np.arange(4), size=2, replace=False)
    # pick a random AF
    minor_ac = np.random.randint(1, 199)
    major_ac = 200 - minor_ac 
    X[nuc_i, snp_i] = np.array([minor_ac, major_ac])

f, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(X, ax=ax)
f.tight_layout()
f.savefig('heatmap.png')

# create edge indices

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

#edge_attributes = torch.tensor()
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

print (data)