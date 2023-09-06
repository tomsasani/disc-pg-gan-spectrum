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
        self.N1 = Parameter(9000, 1000, 20000, "N1")
        self.N2 = Parameter(5000, 1000, 20000, "N2")
        self.T1 = Parameter(2000, 1500, 5000, "T1")
        self.T2 = Parameter(350, 100, 1500, "T2")
        # recombination rate
        self.rho = Parameter(1.25e-8, 1e-9, 1e-7, "rho")
        # mutation rate
        self.mu = Parameter(1.25e-8, 1e-9, 1e-7, "mu")
        # population growth parameter
        self.growth = Parameter(0.01, 0.0, 0.05, "growth")
        # ratio of transitions to transversions (used to paramterize
        # the felsenstein 84 mutation model in msprime)
        self.kappa = Parameter(2.0, 0.1, 3.0, "kappa")
        # gene conversion rate
        self.conversion = Parameter(5e-8, 5e-9, 5e-6, "conversion")
        # gene conversion track length
        self.conversion_length = Parameter(2, 10, 1, "conversion_length")


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
