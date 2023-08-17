"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import math
import msprime

# our imports
import global_vars
import util

################################################################################
# SIMULATION
################################################################################

def simulate_exp(params, sample_sizes, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0,
                                           initial_size=N0,
                                           growth_rate=params.growth.value,),
        msprime.PopulationParametersChange(time=T2,
                                           initial_size=N2,
                                           growth_rate=0),
        msprime.PopulationParametersChange(time=params.T1.value,
                                           initial_size=params.N1.value),
    ]

    ts = msprime.simulate(
        sample_size=sum(sample_sizes),
        demographic_events=demographic_events,
        mutation_rate=params.mu.value,
        length=global_vars.L,
        recombination_rate=params.rho.value,
        random_seed=seed,
    )

    # ts = msprime.sim_ancestry(
    #     samples=sum(sample_sizes),
    #     demography=msprime.Demography.isolated_model(
    #         initial_size=[params.N1.value],
    #         growth_rate=[params.growth.value],
    #     ),
    #     sequence_length=global_vars.L,
    #     recombination_rate=params.rho.value,
    #     random_seed=seed,
    # )

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=msprime.F84(kappa=params.kappa.value),
        random_seed=seed,
    )

    return mts
