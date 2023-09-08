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
    
    model = msprime.MatrixMutationModel(
        global_vars.NUC_ORDER,
        root_distribution=root_dist,
        transition_matrix=norm_transition_matrix,
    )

    return model

################################################################################
# SIMULATION
################################################################################

def simulate_exp(params, sample_sizes, root_distribution, region_length, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N0, growth_rate=params.growth.value,)
    demography.add_population_parameters_change(population="A", time=T2, initial_size=N2, growth_rate=0,)
    demography.add_population_parameters_change(population="A", time=params.T1.value, initial_size=params.N1.value,)


    ts = msprime.sim_ancestry(
        samples=sum(sample_sizes),
        demography=demography,
        sequence_length=region_length,
        recombination_rate=params.rho.value,
        discrete_genome=True,
        random_seed=seed,
    )

    # define mutation model
    mutation_model = parameterize_mutation_model(root_distribution)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        #model=mutation_model,
        random_seed=seed,
        discrete_genome=True,
    )

    return mts

def simulate_isolated(params, sample_sizes, root_distribution, region_length, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    demography = msprime.Demography.isolated_model(initial_size = [params.N1.value], growth_rate = [params.growth.value],)

    ts = msprime.sim_ancestry(
        samples=sum(sample_sizes),
        demography=demography,
        sequence_length=region_length,
        recombination_rate=params.rho.value,
        discrete_genome=True,
        random_seed=seed,
    )

    # define mutation model
    mutation_model = parameterize_mutation_model(root_distribution)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=True,
    )

    return mts

# def simulate_im(params, sample_sizes, seed, reco):
#     """Note this is a 2 population model"""
#     assert len(sample_sizes) == 2

#     # condense params
#     N1 = params.N1.value
#     N2 = params.N2.value
#     T_split = params.T_split.value
#     N_anc = params.N_anc.value
#     mig = params.mig.value

#     # population_configurations = [
#     #     msprime.PopulationConfiguration(sample_size=sample_sizes[0],
#     #         initial_size = N1),
#     #     msprime.PopulationConfiguration(sample_size=sample_sizes[1],
#     #         initial_size = N2)]
    
#     demography = msprime.Demography()
#     demography.add_population(name="A", initial_size=N1)
#     demography.add_population(name="B", initial_size=N2)

#     # no migration initially
#     mig_time = T_split / 2

#     # directional (pulse)
#     if mig >= 0:
#         # migration from pop 1 into pop 0 (back in time)
#         mig_event = msprime.MassMigration(time = mig_time, source = 1,
#             destination = 0, proportion = abs(mig))
#     else:
#         # migration from pop 0 into pop 1 (back in time)
#         mig_event = msprime.MassMigration(time = mig_time, source = 0,
#             destination = 1, proportion = abs(mig))

#     demographic_events = [
#         mig_event,
# 		# move all in deme 1 to deme 0
# 		msprime.MassMigration(
# 			time = T_split, source = 1, destination = 0, proportion = 1.0),
#         # change to ancestral size
#         msprime.PopulationParametersChange(time=T_split, initial_size=N_anc,
#             population_id=0)
# 	]

#     # simulate tree sequence
#     ts = msprime.simulate(
# 		population_configurations = population_configurations,
# 		demographic_events = demographic_events,
# 		mutation_rate = params.mut.value,
# 		length = global_vars.L,
# 		recombination_rate = reco,
#         random_seed = seed)

#     return ts
