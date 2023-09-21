import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import global_vars

n_snps, n_haps, n_nucs = 36, 10, 4

X = np.zeros((n_haps, n_snps, 4))

for snp_i in np.arange(n_snps):
    # pick two random nucleotides
    major_nuc_i, minor_nuc_i = np.random.choice(np.arange(4), size=2, replace=False)
    # pick a random AF
    minor_ac = np.random.randint(1, n_haps - 1)
    major_ac = n_haps - minor_ac 

    # randomly shuffle the haplotype idxs
    hap_idxs = np.arange(n_haps)
    np.random.shuffle(hap_idxs)

    major_haps = hap_idxs[:major_ac]
    minor_haps = hap_idxs[major_ac:]

    X[major_haps, snp_i, major_nuc_i] += 1
    X[minor_haps, snp_i, minor_nuc_i] += 1

X_summed = np.sum(X, axis=0)

# figure out the ref/alt nucleotide indices at each position
node_labels, node_idxs = [], []
for snp_i in np.arange(n_snps):
    minor_i, major_i = np.argsort(X_summed.T[:, snp_i])[-2:]
    node_labels.extend([[major_i], [minor_i]])
    #node_idxs.extend([snp_i, snp_i])
#node_idxs = np.array(node_idxs)

edge_weights = np.zeros(len(global_vars.EDGE_IDXS))
# also need to store haplotype paths through the graph as edge attributes

# loop over snps
snp_j = 1
while snp_j < n_snps:
    snp_i = snp_j - 1
    # figure out the allele states of haplotypes at these two sites
    # i.e., the true reference and alternate alleles
    X_sub = X[:, snp_i:snp_j + 1, :]
    # get ref and alt alleles at snp_i and snp_j
    # first, get the indices of the ref and alt alleles
    # in the node labels
    ref_i, alt_i = snp_i * 2, (snp_i * 2) + 1
    ref_j, alt_j = snp_j * 2, (snp_j * 2) + 1

    # for every path from snp_i to snp_j (ref -> ref, ref -> alt, etc.)
    # figure out how many haplotypes have that path. here, i and j
    # represent the indices in the edge_idxs list. so, if a haplotype
    # matches the reference and alternate allele path given by i -> j,
    # we should increment the edge weight at [i, j] by 1.
    for i, j in ((ref_i, ref_j), (ref_i, alt_j), (alt_i, alt_j), (alt_i, ref_j)):
        # use those indices to get the true ref and alt alleles
        ref, alt = node_labels[i][0], node_labels[j][0]

        # figure out the index of the edge that corresponds to each of these switches
        edge_weight_idx = global_vars.EDGE_IDXS.index([i, j])
        for hap in X_sub:
            # get the path from snp_i to snp_j on this haplotype
            nuc_i = np.where(hap[0] == 1)[0][0]
            nuc_j = np.where(hap[1] == 1)[0][0]
            #print (f"this haplotype has a {}")
            if nuc_i == ref and nuc_j == alt:
                edge_weights[edge_weight_idx] += 1
            
    snp_j += 1


edge_index = torch.tensor(global_vars.EDGE_IDXS, dtype=torch.long)
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

x = torch.tensor(node_labels, dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weights = edge_weights)
data["label"] = 0

g = to_networkx(data, to_undirected=True)

f, ax = plt.subplots()
nx.draw(g, ax=ax)
f.savefig('graph.png')

print (data)