import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import global_vars

def create_graph(X: np.ndarray):

    n_snps, n_haps, n_nucs = X.shape

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = n_snps // 2
    half_S = global_vars.NUM_SNPS // 2

    # instantiate the new region, formatted as (n_haps, n_sites, n_channels)
    region = np.zeros(
        (global_vars.NUM_SNPS, n_haps, 4),
        dtype=np.float32,
    )

    # if we have more than the necessary number of SNPs
    if mid >= half_S:
        # add first channels of mutation spectra
        middle_X = X[mid - half_S:mid + half_S, :, :]
        # sort by genetic similarity
        region[:, :, :] = middle_X

    else:
        other_half_S = half_S + 1 if n_snps % 2 == 1 else half_S
        # use the complete genotype array
        # but just add it to the center of the main array
        region[:, half_S - mid:mid + other_half_S, :-1] = X        


    X_summed = np.sum(region, axis=1)

    # figure out the ref/alt nucleotide indices at each position
    node_labels = []
    for snp_i in np.arange(global_vars.NUM_SNPS):
        minor_i, major_i = np.argsort(X_summed.T[:, snp_i])[-2:]
        node_labels.extend([[major_i], [minor_i]])


    edge_weights = np.zeros(len(global_vars.EDGE_IDXS))

    # loop over snps
    snp_j = 1
    while snp_j < global_vars.NUM_SNPS:
        snp_i = snp_j - 1
        # figure out the allele states of haplotypes at these two sites
        # i.e., the true reference and alternate alleles
        X_sub = region[snp_i:snp_j + 1, :, :]
        #print (X_sub.shape)
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
        for i, j in ((ref_i, ref_j), (ref_i, alt_j), (alt_i, alt_j), (alt_i, ref_j),):
            # use those indices to get the true ref and alt alleles
            ref, alt = node_labels[i][0], node_labels[j][0]
            # figure out the index of the edge that corresponds to each of these switches
            edge_weight_idx = global_vars.EDGE_IDXS.index([i, j])
            for hap in np.transpose(X_sub, (1, 0, 2)):
                #print (hap, hap.shape)
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

    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weights = edge_weights,)
    
    return data
    #g = to_networkx(data, to_undirected=True)

    # f, ax = plt.subplots()
    # nx.draw(g, ax=ax)
    # f.savefig('graph.png')

    # print (data)