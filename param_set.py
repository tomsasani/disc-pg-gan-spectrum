"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from scipy.stats import norm
import sys

class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, value, min, max, name):
        self.value = value
        self.min = min
        self.max = max
        self.name = name
        self.proposal_width = (self.max - self.min) / 15 # heuristic

    def __str__(self):
        s = '\t'.join(["NAME", "VALUE", "MIN", "MAX"]) + '\n'
        s += '\t'.join([str(self.name), str(self.value), str(self.min),
            str(self.max)])
        return s

    def start(self):
        # random initialization
        return np.random.uniform(self.min, self.max)

    def start_range(self):
        """return a range of parameter values, requiring
        the start value to be <= the end value

        Returns:
            _type_: _description_
        """
        start_min = np.random.uniform(self.min, self.max)
        start_max = np.random.uniform(self.min, self.max)
        if start_min <= start_max:
            return [start_min, start_max]
        return self.start_range()

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, curr_value, multiplier):
        if multiplier <= 0: # last iter
            return curr_value
        # normal around current value (make sure we don't go outside bounds)
        new_value = norm(curr_value, self.proposal_width * multiplier).rvs()
        new_value = self.fit_to_range(new_value)
        # if the parameter hits the min or max it tends to get stuck
        if new_value == curr_value or new_value == self.min or new_value == self.max:
            return self.proposal(curr_value, multiplier) # recurse
        else:
            return new_value

    def proposal_range(self, curr_lst, multiplier):
        new_min = self.fit_to_range(norm(curr_lst[0], self.proposal_width *
            multiplier).rvs())
        new_max = self.fit_to_range(norm(curr_lst[1], self.proposal_width *
            multiplier).rvs())
        if new_min <= new_max:
            return [new_min, new_max]
        return self.proposal_range(curr_lst, multiplier) # try again

class ParamSet:

    def __init__(self):

        # population sizes and bottleneck times
        self.N1 = Parameter(9000, 1000, 30000, "N1")
        self.N2 = Parameter(5000, 1000, 30000, "N2")
        self.N3 = Parameter(12000, 1000, 30000, "N3")        
        self.T1 = Parameter(2000, 1500, 5000, "T1")
        self.T2 = Parameter(350, 100, 1500, "T2")
        # recombination rate
        self.rho = Parameter(1.25e-8, 1e-9, 1e-7, "rho")
        # mutation rate
        self.mu = Parameter(1.25e-8, 1e-9, 1e-7, "mu")
        # population growth parameter
        self.growth = Parameter(0.005, 0.0, 0.05, "growth")

        self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
        self.T_split = Parameter(2000, 500, 20000, "T_split")
        self.mig = Parameter(0.05, -0.2, 0.2, "mig")
        # ratio of transitions to transversions (used to paramterize
        # the felsenstein 84 mutation model in msprime)
        # self.kappa = Parameter(2.0, 0.1, 3.0, "kappa")
        # self.conversion = Parameter(5e-8, 5e-9, 5e-6, "conversion")
        # self.conversion_length = Parameter(2, 10, 1, "conversion_length")
        self.mutator_emergence = Parameter(100, 500, 5000, "Tm")
        self.mutator_length = Parameter(10_000, 15_000, 20_000, "Lm")
        self.mutator_start = Parameter(1, 10_000, 30_000, "Ls")
        self.mutator_effect = Parameter(1, 1.5, 2, "Em")

        self.N_gough = Parameter(20_000, 2_000, 40_000, "N_gough")
        self.N_mainland = Parameter(50_000, 15_000, 300_000, "N_mainland")
        self.T_colonization = Parameter(100, 50, 1_500, "T_colonization")
        self.N_colonization = Parameter(1_000, 10, 10_000, "N_colonization")
        self.T_mainland_bottleneck = Parameter(10_000, 3_000, 20_000, "T_mainland_bottleneck")
        self.D_mainland_bottleneck = Parameter(6e-4, 1e-4, 1e-2, "D_mainland_bottleneck")
        self.island_migration_rate = Parameter(8e-4, 0, 1e-3, "island_migration_rate")
        self.mouse_mu = Parameter(6.5e-9, 1e-9, 1e-8, "mouse_mu")
        self.mouse_rho = Parameter(1e-8, 1e-9, 1e-7, "mouse_rho")

    def update(self, names, values):
        """Based on generator proposal, update desired param values"""
        assert len(names) == len(values)

        for j in range(len(names)):
            param = names[j]
            # credit: Alex Pan (https://github.com/apanana/pg-gan)
            attr = getattr(self, param)
            if attr == None:
                sys.exit(param + " is not a recognized parameter.")
            else:
                attr.value = values[j]