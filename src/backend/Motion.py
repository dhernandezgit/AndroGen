import os
import sys
sys.path.append(os.path.abspath('.'))

import random
import numpy as np
from typing import List
from src.backend.common.Distributions import NormalDistribution, UniformDistribution


class Movement:
    def __init__(self, linear_speed, circular_speed, duration):
        self.linear_speed = linear_speed  # UniformDistribution instance
        self.circular_speed = circular_speed  # UniformDistribution instance
        self.duration = duration  # UniformDistribution instance

    def generate_movement(self):
        frames = int(self.duration.random_samples())
        constant_linear_speed = self.linear_speed.random_samples()
        constant_circular_speed = self.circular_speed.random_samples()

        s = [constant_linear_speed] * frames
        w = [constant_circular_speed] * frames
        return s, w


# Updated subclass names with meaningful names
class Stationary(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(0.0, 0.0), UniformDistribution(0, 0), UniformDistribution(9999.0, 9999.0))


class RotatingStationary(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(0.0, 0.0), UniformDistribution(0.01, 0.02), UniformDistribution(10.0, 15.0))


class SlowDrift(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(0.20, 0.8), UniformDistribution(0, 0), UniformDistribution(10.0, 15.0))


class StraightMotion(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(5.0, 8.0), UniformDistribution(0, 0), UniformDistribution(10.0, 15.0))


class GentleCurve(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(5.0, 8.0), UniformDistribution(0.05, 0.15), UniformDistribution(10.0, 15.0))


class TightCurve(Movement):
    def __init__(self):
        super().__init__(UniformDistribution(5.0, 6.0), UniformDistribution(0.15, 0.25), UniformDistribution(10.0, 15.0))


class Motion:
    def __init__(self, movements:List[Movement], duration=25):
        self.movements = movements  # List of Movement instances
        self.duration = duration

    def generate_trajectory(self):
        trajectory_v, trajectory_omega = [], []

        iter = 0
        # Generate movement segments
        while len(trajectory_v) < self.duration:
            movement = self.movements[iter]
            v, omega = movement.generate_movement()
            trajectory_v.extend(v)
            trajectory_omega.extend(omega)
            iter += 1

        # Smooth the entire trajectory
        trajectory_v = self.smooth_transition(trajectory_v[:self.duration])
        trajectory_omega = self.smooth_transition(trajectory_omega[:self.duration])

        return trajectory_v, trajectory_omega

    @staticmethod
    def smooth_transition(values, smoothing_factor=20):
        """Smooth transitions between segments using a moving average."""
        new_values = [values[0]]*int(smoothing_factor/2) + values + [values[-1]]*int(smoothing_factor/2)
        return np.convolve(np.array(new_values).flatten(), np.ones(smoothing_factor) / smoothing_factor, mode='same')
    