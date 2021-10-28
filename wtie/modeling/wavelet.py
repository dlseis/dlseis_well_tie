""" Utilities for creating source wavelets"""
import random
import warnings

import numpy as np

from wtie.processing.spectral import apply_butter_bandpass_filter
from wtie.processing.sampling import Resampler
from wtie.processing.taper import _Taper
from wtie.utils.types_ import FunctionType, List, Tuple



class RandomWaveletCallable:
    """Random seismic source wavelet."""
    def __init__(self,
                 random_base_wavelet_gens: List[FunctionType],
                 perturbations: list=None,
                 resampler: Resampler=None,
                 taper: _Taper=None):
        """
        Parameters
        ----------
        random_base_wavelet_gens : List[FunctionType]?
            List of objects creating random base wavelet when called. See RandomRickerTools.
        transformations : List[BaseTransform], optional
            List of transformations inhereting from BaseTransform. Default is `None`.
        resampler : Resampler, optional
            wtie.processing.sampling.Resampler object
            The default is `None`.
        """


        self.random_base_wavelet_gens = random_base_wavelet_gens
        self.perturbations = perturbations
        self.resampler = resampler
        self.taper = taper

    def __call__(self):
        # random select the wavelet base
        func, func_args, func_kwargs = random.choice(self.random_base_wavelet_gens)()

        # apply perturbations
        return Wavelet(func=func,
                       func_args=func_args,
                       func_kwargs=func_kwargs,
                       perturbations=self.perturbations,
                       resampler=self.resampler,
                       taper=self.taper)


class Wavelet:
    """Random seismic source wavelet."""
    def __init__(self,
                 func: FunctionType,
                 func_args: list=None,
                 func_kwargs: dict=None,
                 perturbations: list=None,
                 resampler: Resampler=None,
                 taper: _Taper=None):

        if func_kwargs is None:
            func_kwargs = {}

        # base wavelet
        self.t_original, self.y_original = func(*func_args, **func_kwargs)

        self._func_name = func.__name__
        #self._args = func_args
        self._args = {key:value for (key,value) in \
                      zip(func.__code__.co_varnames[:len(func_args)], func_args)}

        self._kwargs = func_kwargs

        # random perturbations
        self.perturbations = perturbations
        if perturbations is None:
            self.y_perturbed = np.copy(self.y_original)
        else:
            self.y_perturbed = perturbations(self.y_original)

        # resampling
        if resampler is not None:
            raise DeprecationWarning("Don't use resampler anymore")
            #self.y_perturbed, self.t = resampler(self.y_perturbed, self.t_original)
        #else:
            #self.t = np.copy(self.t_original)
        self.t = np.copy(self.t_original)

        # tapering
        if taper is not None:
            self.y_perturbed = taper(self.y_perturbed)

        # sampling rate
        self.dt_original = self.t_original[1] - self.t_original[0]
        self.dt = self.t[1] - self.t[0]

        # alias
        self.y = self.y_perturbed


    def __str__(self):
        s = "base wavelet: " + str(self._func_name) + "\n"
        s += "num samples: " + str(len(self.y)) + "\n"
        s += "sampling rate: " + str(self.dt) + "\n"
        s += "duration (seconds): " + str(self.dt*len(self.y)) + "\n"
        s += "base args: " + str(self._args) + "\n"
        s += "base kwargs: " + str(self._kwargs) + "\n"
        s += "transformations:\n"
        for line in str(self.perturbations).split("\n")[:-1]:
            s += "\t- " + line + "\n"
        #s +=  str(self.perturbations)
        return s










##################################################
# utils
##################################################

class RandomRickerTools:
    def __init__(self,
                 f_range: Tuple[float, float],
                 dt: float,
                 n_samples: int):

        assert n_samples % 2 == 0, "Best specify an even number of samples..."

        self.f_range = f_range
        self.dt = dt
        self.n_samples = n_samples


    def __call__(self):
        # returns function, args, kwargs
        f = np.random.uniform(self.f_range[0], self.f_range[1])
        return ricker, (f, self.dt, self.n_samples), None


class RandomOrmbyTools:
    def __init__(self,
                 f0_range: Tuple[float, float],
                 f1_range: Tuple[float, float],
                 f2_range: Tuple[float, float],
                 f3_range: Tuple[float, float],
                 dt: float,
                 n_samples: int):

        assert n_samples % 2 == 0, "Best specify an even number of samples..."

        self.f0_range = f0_range
        self.f1_range = f1_range
        self.f2_range = f2_range
        self.f3_range = f3_range
        self.dt = dt
        self.n_samples = n_samples

        _warn_nyquist(f3_range[-1], dt)


    def __call__(self):
        # returns function, args, kwargs
        Df = 9

        f0 = np.random.uniform(self.f0_range[0], self.f0_range[1])

        f1_min = max(self.f1_range[0], f0+Df)
        f1 = np.random.uniform(f1_min, max(self.f1_range[1], f1_min))

        f2_min = max(self.f2_range[0], f1+Df)
        f2 = np.random.uniform(f2_min, max(self.f2_range[1], f2_min))

        f3_min = max(self.f3_range[0], f2+Df)
        f3 = np.random.uniform(f3_min, max(self.f3_range[1], f3_min))
        return ormby, ((f0, f1, f2, f3), self.dt, self.n_samples), None



class RandomButterworthTools:
    def __init__(self,
                 lowcut_range: Tuple[float, float],
                 highcut_range: Tuple[float, float],
                 dt: float,
                 n_samples: int,
                 order_range: Tuple[float, float]=(6,6)):

        self.lowcut_range = lowcut_range
        self.highcut_range = highcut_range
        self.dt = dt
        self.n_samples = n_samples
        self.order_range = order_range


    def __call__(self):
        # retunrs function, args, kwargs
        lowcut = np.random.uniform(self.lowcut_range[0], self.lowcut_range[1])
        highcut = np.random.uniform(self.highcut_range[0], self.highcut_range[1])
        order = np.random.randint(self.order_range[0], self.order_range[1]+1)
        return butterworth, (lowcut, highcut, self.dt, self.n_samples), dict(order=order)





def ricker(f: float, dt: float, n_samples: int):
    #n_samples = int(round(duration / dt))
    duration = n_samples * dt
    t = np.arange(-duration/2, (duration-dt)/2, dt)
    #t = np.arange(-duration/2, duration/2, dt)

    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y


def ormby(f: Tuple[float], dt: float, n_samples: int):

    assert len(f) == 4, "You need to specify 4 frequencies for the trapezoid definition"
    assert 0 < f[0] < f[1] < f[2] < f[3], "Frequencies must be strictly increasing"
    f0, f1, f2, f3 = f

    duration = n_samples * dt
    #t = np.arange(-duration/2, (duration-dt)/2, dt)
    t = np.arange(-duration/2, duration/2, dt)

    def g(f, t):
        return (np.sinc(f * t)**2) * ((np.pi * f) ** 2)

    pf32 = (np.pi * f3) - (np.pi * f2)
    pf10 = (np.pi * f1) - (np.pi * f0)

    w = ((g(f3, t)/pf32) - (g(f2, t)/pf32) -
         (g(f1, t)/pf10) + (g(f0, t)/pf10))

    w /= np.max(w)
    return t, w




def butterworth(lowcut: float,
                highcut: float,
                dt: float,
                n_samples: int,
                order: int=8,
                rescale: bool=True,
                zero_phase=True,
                ) -> np.ndarray:

    # for edge effects
    n_samples_tmp = 3*n_samples

    spike = np.zeros((n_samples_tmp,), dtype=float)
    spike[n_samples_tmp//2] = 1.

    fs = 1/dt

    duration = n_samples * dt
    t = np.arange(-duration/2, (duration-dt)/2, dt)



    y = apply_butter_bandpass_filter(spike, lowcut, highcut, fs,
                                        order=order, zero_phase=zero_phase)

    y = y[n_samples:-n_samples]

    assert t.size == n_samples == y.size

    if rescale:
        y /= max(y.max(), -y.min())

    return t, y



#########################
# utils
#########################
def _warn_nyquist(f: float, dt: float):
    # assumes Hertz and seconds
    fN = 1/(2*dt)
    if f > fN:
        warnings.warn("%.1f Hz greater than Nyquist frequency (%.1f Hz)" % (f, fN))