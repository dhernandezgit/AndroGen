import os
import numpy as np
from scipy.stats import uniform, norm

class UniformDistribution:
    def __init__(self, a=0, b=1):
        """
        Initialize a Uniform distribution.
        :param a: Lower bound (inclusive)
        :param b: Upper bound (exclusive)
        """
        self.a = a
        self.b = b

    def random_samples(self, n=1):
        """Generate n random samples."""
        if n == 1:
            return np.random.uniform(self.a, self.b)
        return np.random.uniform(self.a, self.b, n)


class NormalDistribution:
    def __init__(self, mean=0, std=1):
        """
        Initialize a Normal (Gaussian) distribution.
        :param mean: Mean of the distribution
        :param std: Standard deviation of the distribution
        """
        self.mean = mean
        self.std = std

    def random_samples(self, n=1):
        """Generate n random samples."""
        if n == 1:
            return np.random.normal(self.mean, self.std)
        return np.random.normal(self.mean, self.std, n)