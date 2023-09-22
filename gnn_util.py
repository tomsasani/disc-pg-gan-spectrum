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
    node_labels = np.zeros(global_vars.NUM_SNPS * 2)
    for snp_i in np.arange(global_vars.NUM_SNPS):
        minor_i, major_i = np.argsort(X_summed.T[:, snp_i])[-2:]
        node_labels[snp_i * 2] = major_i
        node_labels[snp_i * 2 + 1] = minor_i
        #node_labels.extend([[major_i], [minor_i]])

    edge_weights = np.zeros(len(global_vars.EDGE_IDXS))

    # for every haplotype, we want to figure out its "path" 
    # through the graph so that we can update the edge weights
    # for each edge in the graph
    for hap_i in np.arange(n_haps):
        hap_m = region[:, hap_i, :]
        # first, get the indices on each haplotype at each SNP where 
        # the haplotype has an ALT allele
        snp_i, nuc_i = np.where(hap_m > 0)
        # we can also figure out the true major allele at every
        # position snp_i
        major_alleles = node_labels[snp_i * 2]
        # on each haplotype at each snp index, we can figure out whether
        # the haplotype has the major or minor allele.
        # if is_major == 0, then by default, is_minor == 1, and
        # vice versa 
        is_major = 1 - (nuc_i == major_alleles)
        # using the information about whether the haplotype has the
        # major (1) or minor (0) allele at this site, we can now figure
        # out what the value of this haplotype at this site should be
        # in the first element of the EDGE_INDEX tuple. that is, 
        # if the haplotype at this position has the major allele,
        # then its EDGE_INDEX would start with snp_i * 2. if the
        # haplotype at this position has the minor allele, then its
        # edge index would start with snp_i * 2 + 1.
        edge_index_i = (snp_i * 2) + is_major
        # now we can zip up the consecutive edge indices into EDGE_INDEX tuples
        tuples = zip(edge_index_i[:-1], edge_index_i[1:])
        for t in tuples:
            edge_weights[global_vars.EDGE_IDXS.index(list(t))] += 1
        
    # # now that we know the element of the EDGE INDEX tuple at every
    # # position on every haplotype, we can count up all of the instances
    # # of each EDGE_INDEX tuple across every pair of sites.

    # # we start at snp_i == 0, snp_j == 1
    # j = 1
    # while j < (global_vars.NUM_SNPS):
    #     i = 0
    #     # figure out where the snp indices match
    #     snp_i_idxs = np.where(snp_i == i)[0]
    #     snp_j_idxs = np.where(snp_i == j)[0]
    #     print (snp_i_idxs)
    #     # get the edge indices of all haplotypes
    #     # at snp_i and snp_j
    #     edge_tuples = list(zip(edge_index_i[snp_i_idxs], edge_index_i[snp_j_idxs]))
    #     print (edge_tuples)
    #     break
    


    # print (snp_i.shape, hap_i.shape, nuc_i.shape)

    # # loop over snps
    # snp_j = 1
    # while snp_j < global_vars.NUM_SNPS:
    #     snp_i = snp_j - 1
    #     # figure out the allele states of haplotypes at these two sites
    #     # i.e., the true reference and alternate alleles
    #     X_sub = region[snp_i:snp_j + 1, :, :]
    #     #print (X_sub.shape)
    #     # get ref and alt alleles at snp_i and snp_j
    #     # first, get the indices of the ref and alt alleles
    #     # in the node labels
    #     ref_i, alt_i = snp_i * 2, (snp_i * 2) + 1
    #     ref_j, alt_j = snp_j * 2, (snp_j * 2) + 1

    #     # for every path from snp_i to snp_j (ref -> ref, ref -> alt, etc.)
    #     # figure out how many haplotypes have that path. here, i and j
    #     # represent the indices in the edge_idxs list. so, if a haplotype
    #     # matches the reference and alternate allele path given by i -> j,
    #     # we should increment the edge weight at [i, j] by 1.
    #     for i, j in ((ref_i, ref_j), (ref_i, alt_j), (alt_i, alt_j), (alt_i, ref_j),):
    #         # use those indices to get the true ref and alt alleles
    #         ref, alt = node_labels[i][0], node_labels[j][0]
    #         # figure out the index of the edge that corresponds to each of these switches
    #         edge_weight_idx = global_vars.EDGE_IDXS.index([i, j])
    #         for hap in np.transpose(X_sub, (1, 0, 2)):
    #             #print (hap, hap.shape)
    #             # get the path from snp_i to snp_j on this haplotype
    #             nuc_i = np.where(hap[0] == 1)[0][0]
    #             nuc_j = np.where(hap[1] == 1)[0][0]
                
    #             #print (f"this haplotype has a {}")
    #             if nuc_i == ref and nuc_j == alt:
    #                 edge_weights[edge_weight_idx] += 1
                
    #     snp_j += 1

    edge_index = torch.tensor(global_vars.EDGE_IDXS, dtype=torch.long)
    edge_weights = torch.tensor([[e] for e in edge_weights], dtype=torch.float)
    x = torch.tensor([[n] for n in node_labels], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weights = edge_weights,)

    return data
    #g = to_networkx(data, to_undirected=True)

    # f, ax = plt.subplots()
    # nx.draw(g, ax=ax)
    # f.savefig('graph.png')

    # print (data)