"""Tapering utilities."""

import numpy as np

from wtie.utils.types_ import _size_2_t



class _Taper:
    """Base class, all implementations should inherit from. Do not instanciate."""
    def __init__(self, size: _size_2_t):
        """
        Parameters
        ----------
        size : int or tuple(int,int)
            Size of the linear ramp from both ends. If a single number is
            provided the taper will be symmetric.
        """
        if type(size) is int:
            size = (size, size)
        self.size = size

        # overwriten in children class
        self.left_tapper = None
        self.right_tapper = None


    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray
            1D input data.

        Returns
        -------
        Inplace tapering of the input data.
        """
        data[:self.size[0]] *= self.left_tapper
        data[-self.size[1]:] *= self.right_tapper
        return data



class SoftTaper(_Taper):
    def __init__(self, size: _size_2_t, plateau: float=0.5):
        super().__init__(size)

        assert 0 < plateau < 1

        self.left_tapper = (self.left_tapper + plateau) / (1 + plateau)
        self.right_tapper = (self.right_tapper + plateau) / (1 + plateau)




class LinearTaper(_Taper):
    """1-dimensional linear taper."""
    def __init__(self, size: _size_2_t):
        super().__init__(size)

        self.left_tapper = np.linspace(0., 1., num=self.size[0])
        self.right_tapper = np.linspace(1., 0., num=self.size[1])




class CosinePowerTaper(_Taper):
    """1-dimensional cosine square taper."""
    def __init__(self, size: _size_2_t, power: int):
        super().__init__(size)

        self.power = power

        t_left = np.deg2rad(np.linspace(0.,90.,num=self.size[0]))[::-1]
        t_right = np.deg2rad(np.linspace(0.,90.,num=self.size[1]))

        self.left_tapper = np.cos(t_left)**power
        self.right_tapper = np.cos(t_right)**power



class CosSquareTaper(CosinePowerTaper):
    def __init__(self, size: _size_2_t):
        super().__init__(size, 2)










