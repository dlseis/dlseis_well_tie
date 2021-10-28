"""Utilities to generate reflectivity series."""
import random

import numpy as np

from wtie.modeling.noise import open_simplex_noise
from wtie.utils.types_ import List, Tuple


class RandomReflectivityCallable:
    """Random seismic source wavelet."""
    def __init__(self,
                 random_reflectivity_choosers: List["RandomReflectivityChooser"]):
        """ """


        self.random_reflectivity_choosers = random_reflectivity_choosers


    def __call__(self):
        return random.choice(self.random_reflectivity_choosers)()

    def __str__(self):
        s = ""
        for f in self.random_reflectivity_choosers:
            s += (f.__class__.__name__ + "\n")
        return s





############################
# Random parameters
############################
class RandomReflectivityChooser:
    """TODO: common abstarct base class for reflectivity choosers. (see ABC)"""
    pass




class RandomSpikeReflectivity:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples


    def __call__(self):
        spike = np.zeros((self.num_samples,))
        idx = np.random.randint(0, self.num_samples)
        amp = np.random.uniform(0.5,1.0)
        spike[idx] = amp
        return spike


class RandomWeakUniformReflectivityChooser(RandomReflectivityChooser):
    def __init__(self,
                 num_samples: int,
                 sparsity_rate_range: Tuple[float, float],
                 max_amplitude_range: Tuple[float, float]
                 ):


        self.sparsity_rate_range = sparsity_rate_range
        self.max_amplitude_range = max_amplitude_range
        self.num_samples = num_samples


    def __call__(self):
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        max_ = np.random.uniform(self.max_amplitude_range[0], self.max_amplitude_range[1])
        ref = RandomUniformReflectivity(self.num_samples, sparsity_rate=sr)()
        ref *= max_ #assumes _max in ]0,1[ and ref is normalized to [-1,1]

        return ref


class RandomUniformReflectivityChooser(RandomReflectivityChooser):
    def __init__(self,
                 num_samples: int,
                 sparsity_rate_range: Tuple[float, float],
                 ):


        self.sparsity_rate_range = sparsity_rate_range
        self.num_samples = num_samples


    def __call__(self):
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        return RandomUniformReflectivity(self.num_samples, sparsity_rate=sr)()



class RandomSimplexReflectivityChooser(RandomReflectivityChooser):
    def __init__(self,
                 num_samples: int,
                 sparsity_rate_range: Tuple[float, float],
                 variation_scale_range: Tuple[int, int]
                 ):


        self.sparsity_rate_range = sparsity_rate_range
        self.variation_scale_range = variation_scale_range
        self.num_samples = num_samples


    def __call__(self):
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        vs = np.random.uniform(self.variation_scale_range[0], self.variation_scale_range[1])
        return RandomSimplexReflectivity(self.num_samples, sparsity_rate=sr, variation_scale=vs)()


class RandomBiUniformReflectivityChooser(RandomReflectivityChooser):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __call__(self):
        return RandomBiUniformReflectivity(self.num_samples)()


############################
# Base reflectivity creation
#############################
class SpikeReflectivity:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

        spike = np.zeros((self.num_samples,))
        spike[spike.size//2] = 1.

        self.spike = spike

    def __call__(self):
        return self.spike


class RandomUniformReflectivity:
    """Create random 1D reflectivity series. Returns values between -1.0 and 1.0."""


    def __init__(self,
                 num_samples: int,
                 sparsity_rate: float = .6
                 ):
        """
        Parameters
        ----------
        num_samples : int
            Length (number of samples) of the reflectivity series.
        sparsity_rate : int, optional
            Parameter controling the sparsity of the reflectivity. Value should
            be between 0 and 1. Larger sparsity_rate yields more zeros.
            The default is .6.
        power_stretch : int, optional - DISABLED
            Rasing the reflectivity to given power to strech the amplitde distribution.

        """
        assert (sparsity_rate >=0. and sparsity_rate <= 1.)
        self.n = num_samples
        self.sparsity_rate = sparsity_rate
        #self.power_stretch = power_stretch


    def __call__(self) -> np.ndarray:
        """
        Returns
        -------
        A numpy ndarray of type float with shape (num_samples,)
        """
        # generate random response in [0, 1)
        reflectivity = np.random.rand(self.n)


        # zeroing
        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate
        reflectivity[zeros] = 0.

        # strech
        #reflectivity **= self.power_stretch


        # adjust sign
        sign = np.random.rand(self.n)
        reflectivity[sign > 0.5] *= -1

        return reflectivity







class RandomSimplexReflectivity:
    """Create simplex 1D reflectivity series. Returns values between -1.0 and 1.0.
    TODO: work in progress...
    """


    def __init__(self,
                 num_samples: int,
                 sparsity_rate: float = .6,
                 variation_scale: float = 150.
                ):
        """
        See docstring of RandomUniformReflectivity
        """
        assert (sparsity_rate >=0. and sparsity_rate <= 1.)

        self.n = num_samples
        self.sparsity_rate = sparsity_rate
        self.variation_scale = variation_scale

        #self.get_uniform_sparse_series = RandomUniformReflectivity(num_samples=num_samples,
                                                         #sparsity_rate=sparsity_rate)




    def __call__(self) -> np.ndarray:
        """
        Returns
        -------
        A numpy ndarray of type float with shape (num_samples,)
        """
        # random uniform sparse positive series
        reflectivity = open_simplex_noise(size=self.n, amplitude_scale=1,
                                          octaves=4, base=None, variation_scale=self.variation_scale)
        #reflectivity = normalize(reflectivity, a=-1.,b=1.)
        reflectivity -= reflectivity.mean()

        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate
        reflectivity[zeros] = 0.

        return reflectivity


class RandomBiUniformReflectivity:
    """Create random 1D reflectivity series. Returns values between -1.0 and 1.0."""


    def __init__(self,
                 num_samples: int,
                 sparsity_rate_big: float = .95,
                 sparsity_rate_small: float = .5,
                 big_reflectivity_min: float = 0.6,
                 small_reflcetivity_max: float = 0.2
                 ):

        assert (sparsity_rate_big >=0. and sparsity_rate_big <= 1.)
        assert (sparsity_rate_small >=0. and sparsity_rate_small <= 1.)

        self.n = num_samples
        self.sparsity_rate_big = sparsity_rate_big
        self.sparsity_rate_small = sparsity_rate_small
        self.big_reflectivity_min = big_reflectivity_min
        self.small_reflcetivity_max = small_reflcetivity_max


    def __call__(self) -> np.ndarray:
        """
        Returns
        -------
        A numpy ndarray of type float with shape (num_samples,)
        """
        # generate random response in [0, 1)
        reflectivity_big = np.random.uniform(low=self.big_reflectivity_min,
                                             high=1.0, size=(self.n,))
        reflectivity_small = np.random.uniform(low=0,
                                               high=self.small_reflcetivity_max,
                                               size=(self.n,))



        # zeroing
        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate_big
        reflectivity_big[zeros] = 0.

        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate_small
        reflectivity_small[zeros] = 0.


        # merge
        reflectivity = np.maximum(reflectivity_big, reflectivity_small)


        # adjust sign
        sign = np.random.rand(self.n)
        reflectivity[sign > 0.5] *= -1

        return reflectivity
