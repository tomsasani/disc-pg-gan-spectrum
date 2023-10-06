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
import param_set
import matplotlib.pyplot as plt
import demesdraw

def get_transition_matrix():

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
            ri, ai = global_vars.NUC2IDX[r], global_vars.NUC2IDX[a]
            transition_matrix[ri, ai] = prob

    # normalize transition matrix so that rows sum to 1
    rowsums = np.sum(transition_matrix, axis=1)
    norm_transition_matrix = transition_matrix / rowsums[:, np.newaxis]
    np.fill_diagonal(norm_transition_matrix, val=0)

    return norm_transition_matrix

def parameterize_mutation_model(root_dist: np.ndarray):

    norm_transition_matrix = get_transition_matrix()

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

def simulate_exp(params, sample_sizes, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    N0 = params.N2.value / math.exp(-params.growth.value * params.T2.value)

    demography = msprime.Demography()
    # at present moment, create population A with the size it should be
    # following its period of exponential growth
    demography.add_population(
        name="A",
        initial_size=N0,
        growth_rate=params.growth.value,
    )
    # T2 generations in the past, change the population size to be N2
    demography.add_population_parameters_change(
        population="A",
        time=params.T2.value,
        initial_size=params.N2.value,
        growth_rate=0,
    )

    # T1 generations in the past, change the population size to be N1
    demography.add_population_parameters_change(
        population="A",
        time=params.T1.value,
        initial_size=params.N1.value,
        #growth_rate=0,
    )

    # sample sample_sizes monoploid haplotypes from the diploid population
    ts = msprime.sim_ancestry(
        #samples=sum(sample_sizes),
        samples = [msprime.SampleSet(sum(sample_sizes), ploidy=1)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.value,
        discrete_genome=False, # ensure no multi-allelics
        random_seed=seed,
        ploidy=2,
    )

    # define mutation model
    # mutation_model = parameterize_mutation_model(root_dist)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        #model=msprime.JC69(state_independent=False),
        #model=msprime.BinaryMutationModel(), # ensure no silent
        random_seed=seed,
        discrete_genome=False,
    )

    return mts

def simulate_im(params, sample_sizes, seed):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # condense params
    N1 = params.N1.value
    N2 = params.N2.value
    T_split = params.T_split.value
    N_anc = params.N_anc.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = N1),
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = N2)]

    # no migration initially
    mig_time = T_split/2

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
    # move all in deme 1 to deme 0
    msprime.MassMigration(
    time = T_split, source = 1, destination = 0, proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T_split, initial_size=N_anc,
            population_id=0)
    ]

    # simulate tree sequence
    ts = msprime.simulate(
    population_configurations = population_configurations,
    demographic_events = demographic_events,
    mutation_rate = params.mu.value,
    length = global_vars.L,
    recombination_rate = params.rho.value,
        random_seed = seed)

    return ts

def simulate_gough(params, sample_sizes, seed):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # N_gough = params.N_bottleneck.value / math.exp(-params.growth.value * params.T_bottleneck.value)

    # size = n_bot / e^(-g * t_bot)

    # size * (e^(-g * t_bot)) = n_bot

    # e ^ (-g * t_bot) = n_bot / size

    # -g * t_bot = ln(n_bot / size)

    # -g = ln(n_bot / size) / t_bot

    # g = -1 * ln(n_bot / size) / t_bot

    # calculate growth rate given colonization Ne and current Ne
    gough_growth = -1 * np.log(params.N_colonization.value / params.N_gough.value) / params.T_colonization.value

    demography = msprime.Demography()
    # at present moment, gough population which has grown exponentially
    demography.add_population(
        name="gough",
        initial_size=params.N_gough.value,
        growth_rate=gough_growth,
    )
    demography.add_population(
        name="mainland",
        initial_size=params.N_mainland.value,
        growth_rate=0,
    )
    demography.add_population(
        name="ancestral",
        initial_size=params.N_mainland.value,
        growth_rate=0,
    )
    demography.set_migration_rate(
        source="gough",
        dest="mainland",
        rate=params.island_migration_rate.value,
    )
    demography.add_population_split(
        time=params.T_colonization.value,
        derived=["gough", "mainland"],
        ancestral="ancestral",
    )

    # demography.add_instantaneous_bottleneck(
    #     time=params.T_mainland_bottleneck.value,
    #     population="ancestral",
    #     strength=params.D_mainland_bottleneck.value
    # )

    # graph = msprime.Demography.to_demes(demography)
    # f, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
    # demesdraw.tubes(graph, ax=ax, seed=1)
    # f.savefig('gough_demography.png', dpi=200)

    # sample sample_sizes monoploid haplotypes from the diploid population
    ts = msprime.sim_ancestry(
        #samples=sum(sample_sizes),
        samples=[
            msprime.SampleSet(sample_sizes[0], population="gough", ploidy=2),
            msprime.SampleSet(sample_sizes[1], population="mainland", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.value,
        discrete_genome=False,  # ensure no multi-allelics
        random_seed=seed,
        ploidy=2,
    )

    # define mutation model
    # mutation_model = parameterize_mutation_model(root_dist)

    mts = msprime.sim_mutations(
        ts,
        rate=params.mouse_mu.value,
        #model=msprime.JC69(state_independent=False),
        #model=msprime.BinaryMutationModel(), # ensure no silent
        random_seed=seed,
        discrete_genome=False,
    )

    return mts

if __name__ == "__main__":
    params = param_set.ParamSet()

    simulate_gough(params, [30, 30], 4242)