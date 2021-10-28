"""Sampling utilities."""

import numpy as np
from scipy.signal import resample
from scipy.signal import decimate as _decimate


from wtie.utils.types_ import Tuple



class Resampler:
    """1-dimensional linear resampling."""

    def __init__(self, current_dt: float, resampling_dt: float):
        """
        Parameters
        ----------
        current_dt : float
            Current sampling rate [s]
        resampling_dt : float
            New sampling rate [s]
        """

        raise NotImplementedError()

        # division factor: n_samples_new = n_samples_old // div_factor
        self.div_factor = resampling_dt / current_dt


    def __call__(self, signal: np.ndarray, t: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray] :
        n_samples_new = int(len(signal) // self.div_factor)
        signal_resamp, t_resamp = resample(signal, num=n_samples_new, t=t)
        return signal_resamp, t_resamp



def downsample(s: np.ndarray, div_factor: int) -> np.ndarray:
    assert div_factor > 1
    # lowpass and decimate
    signal_resamp = _decimate(s, div_factor)
    # correct for DC bias
    signal_resamp = signal_resamp - signal_resamp.mean() + s.mean()
    return signal_resamp