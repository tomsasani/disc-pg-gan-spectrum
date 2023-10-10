import msprime
import tensorflow as tf
import sklearn
import math
import numpy as np

import param_set
import global_vars
import generator

def simulate(params, sample_sizes, seed):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    N0 = params.N2.value / math.exp(-params.growth.value * params.T2.value)

    demography = msprime.Demography()
    # at present moment, create population A with the size it should be
    # following its period of exponential growth
    demography.add_population(name="A", initial_size=N0, growth_rate=params.growth.value)
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
        growth_rate=0,
    )

    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(sum(sample_sizes), population="A", ploidy=1)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.value,
        discrete_genome=True,
        random_seed=seed,
        ploidy=2
    )

    # define RateMap
    mutator_start_pos = params.mutator_start.value
    mutator_interval_length = params.mutator_length.value
    ratemap = msprime.RateMap(
        position=[
            0, mutator_start_pos, mutator_start_pos + mutator_interval_length,
            global_vars.L
        ],
        rate=[
            params.mu.value, params.mu.value * params.mutator_effect,
            params.mu.value
        ],
    )

    # simulate mutation rate up until specified time with normal mutation rate
    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        random_seed=seed,
        discrete_genome=False,
        start_time = params.mutator_emergence.value
    )

    # after mutator emergence, simulate higher mutation rate in a unique region
    mts = msprime.sim_mutations(
        mts,
        rate=ratemap,
        random_seed=seed,
        discrete_genome=False,
        end_time=params.mutator_emergence.value,
    )

    return mts

parameters = param_set.ParamSet()

# set inferred parameter values to the values inferred using the GAN
expected_params = [21_705, 3_581, 4_269, 899, 5.67e-3]

# define a Generator object with the above simulator
# NOTE: we can only update parameter values for the parameter names specified
# when we initialize the generator. so, in order to make sure the N1, N2 etc.
# parameter values are always the "inferred" values, we have to redudnantly
# include those names in the initialization
generator = generator.Generator(simulate, ["N1", "N2", "T1", "T2", "growth", "mutator_emergence", "mutator_length", "mutator_start", "mutator_effect"], [200], 42)

# first, simulate a batch of 10_000 "normal" regions with normal mutaiton rates
# and inferred parameters
BATCH_SIZE = 10_000

# for normal batches, we pass in some benign values for the mutator effect
mutator_params = [1_000, 1_000, 1_000, 1.]
normal_batches = generator.simulate_batch(batch_size = BATCH_SIZE, params = expected_params + mutator_params)

training_batches = np.zeros(
            (BATCH_SIZE, 200, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
            dtype=np.float32)

# define parameter objects for each of the mutator variables
mutator_emergence = param_set.Parameter(100, 500, 5000, "Tm")
mutator_length = param_set.Parameter(10_000, 15_000, 20_000, "Lm")
mutator_start = param_set.Parameter(1, 10_000, 30_000, "Ls")
mutator_effect = param_set.Parameter(1, 1.5, 2, "Em")

mutator_params = [mutator_emergence, mutator_length, mutator_start, mutator_effect]

for batch in range(BATCH_SIZE):
    # random parameters for the mutator variables
    random_param_vals = [p.start() for p in mutator_params]

    normal_batch = generator.simulate_batch(batch_size = 1, params = expected_params + random_param_vals)
    training_batches[batch] = normal_batch

combined_data = np.concatenate(training_batches, normal_batches, axis=0)
combined_labels = np.tile([1, 0], BATCH_SIZE)

disc = tf.saved_model.load("saved_model/3262024358/fingerprint.pb")
disc_recon = discriminator.OnePopModel(200, saved_model=disc)

X_train, X_test, y_train, y_test = train_test_split(combined_data,
                                                    combined_labels,
                                                    test_size=0.2,
                                                    random_state=42)

history = disc_recon.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))