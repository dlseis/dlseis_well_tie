"""Noise models."""

import numpy as np
from noise import pnoise1

from wtie.utils.types_ import Tuple

class NoiseCallable:
    """TODO: common abstarct base class for noise models. (see ABC)"""
    pass

class RandomWhiteNoise(NoiseCallable):
    def __init__(self, size: int, scale_range: Tuple[float, float]):
        self.size = size
        self.scale_range = scale_range

    def __call__(self):
        std = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return white_noise(self.size, std)


class WhiteNoise(NoiseCallable):
    def __init__(self, size: int, scale: float):
        self.size = size
        self.scale = scale

    def __call__(self):
        return white_noise(self.size, self.scale)



##################################
# UTILS FUNCTIONS
##################################

def white_noise(size: int, scale: float, loc: float=0.0):
    """Returns 1D Gaussian noise."""
    return np.random.normal(loc=loc, scale=scale, size=size)

def open_simplex_noise(size: int,
                       amplitude_scale: float,
                       variation_scale: float=1.0,
                       octaves: int=6,
                       base_order: int=4,
                       base: int=None):
    """Returns Perlin like noise
    TODO: probably not correct..."""
    y = np.zeros((size,), dtype=float)

    if base is None:
        b = 10**base_order
        base = np.random.randint(low=-b,high=b)

    for i in range(size):
        x = float(i)/ (size / variation_scale)
        y[i] = pnoise1(x+base, octaves=octaves)

    if np.random.randint(2) == 0:
        y = y[::-1]

    if np.random.randint(2) == 0:
        y *= -1.0

    y /= max(y.max(),-y.min())
    y *= amplitude_scale

    if np.random.randint(2) == 0:
        y = y[::-1]

    return y