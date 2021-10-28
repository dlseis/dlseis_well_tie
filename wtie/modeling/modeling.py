"""Modeling."""

import warnings

import numpy as np



class ModelingCallable:
    """TODO: common abstarct base class for modeling. (see ABC)"""
    pass


class ConvModeler(ModelingCallable):
    def __init__(self, kwargs: dict=None):
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

    def __call__(self, wavelet, reflectivity, noise=None):
        return convolution_modeling(wavelet, reflectivity, noise, **self.kwargs)






######################################
# UTILS FUNCTIONS
######################################

def convolution_modeling(wavelet: np.ndarray,
                         reflectivity: np.ndarray,
                         noise: np.ndarray=None,
                         mode: str='same'
                         ) -> np.ndarray:
    """Simple convolutional model:
    s(t) = w(t) * r(t) + n(t)
    seismic = wavelet * reflectivity + noise
    basis: time in seconds
    """


    if wavelet.size % 2 == 0:
        wavelet = np.concatenate((wavelet, np.zeros((1,))))
    else:
        warnings.warn("Wavelet has odd number of samples.")

    trace = np.convolve(wavelet, reflectivity, mode=mode)

    if wavelet.size > reflectivity.size:
        warnings.warn("Wavelet has more samples than relfectivity.")
        diff_idx = trace.size - reflectivity.size
        trace = trace[diff_idx//2:-diff_idx//2]

    if noise is not None:
        trace += noise

    return trace











