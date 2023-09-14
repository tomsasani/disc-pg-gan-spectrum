"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import math
import msprime
import numpy as np
from typing import List

# our imports
import global_vars
import util

ALLELE2IDX = dict(zip(global_vars.NUC_ORDER, range(len(global_vars.NUC_ORDER))))

def parameterize_mutation_model(root_dist: np.ndarray):

    # define expected mutation probabilities
    mutations = ["C>T", "C>A", "C>G", "A>T", "A>C", "A>G"]
    lambdas = np.array([0.25, 0.1, 0.1, 0.1, 0.1, 0.35])

    transition_matrix = np.zeros((4, 4))

    # for every mutation type...
    for mutation, prob in zip(mutations, lambdas):
        # get the indices of the reference and alt alleles
        ref, alt = mutation.split(">")
        # as well as the reverse complement
        ref_rc, alt_rc = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
        # add its mutation probability to the transition matrix
        for r, a in ((ref, alt), (ref_rc, alt_rc)):
            ri, ai = ALLELE2IDX[r], ALLELE2IDX[a]
            transition_matrix[ri, ai] = prob

    # normalize transition matrix so that rows sum to 1
    rowsums = np.sum(transition_matrix, axis=1)
    norm_transition_matrix = transition_matrix / rowsums[:, np.newaxis]
    np.fill_diagonal(norm_transition_matrix, val=0)

    if np.sum(root_dist) != 1:
        #print ("Root distribution doesn't sum to 1!", root_dist)
        root_dist = root_dist / np.sum(root_dist)

    model = msprime.MatrixMutationModel(
        global_vars.NUC_ORDER,
        root_distribution=root_dist,
        transition_matrix=norm_transition_matrix,
    )

    return model

################################################################################
# SIMULATION
################################################################################

def simulate_exp(params, sample_sizes, root_dist, region_len, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0,
                                           initial_size=N0,
                                           growth_rate=params.growth.value),
        msprime.PopulationParametersChange(time=T2,
                                           initial_size=N2,
                                           growth_rate=0),
        msprime.PopulationParametersChange(time=params.T1.value,
                                           initial_size=params.N1.value),
    ]

    demography = msprime.Demography.from_old_style(demographic_events=demographic_events)

    ts = msprime.sim_ancestry(
        samples=sum(sample_sizes),
        demography=demography,
        sequence_length=region_len,
        recombination_rate=params.rho.value,
        discrete_genome=True,
        random_seed=seed,
    )

    # define mutation model
    mutation_model = parameterize_mutation_model(root_dist)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=True,
    )

    return mts
