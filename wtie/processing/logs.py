"""Some functions to pre-rpocess the logs."""

import numpy as np
import pandas as pd

from numba import njit

from scipy.signal import medfilt
#from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter1d
#from scipy.stats import hmean




def despike(data: np.ndarray, median_size: int=31, threshold: float=1.,
            xmin_clip:float=None, xmax_clip:float=None) -> np.ndarray:

    data = np.copy(data)
    med = medfilt(np.copy(data),median_size)
    noise = np.abs(data - med)
    threshold = threshold * np.nanstd(noise)
    mask = np.abs(noise) > threshold
    data[mask] = np.nan
    if xmin_clip: data[data<xmin_clip] = np.nan
    if xmax_clip: data[data>xmax_clip] = np.nan
    return data


def interpolate_nans(x: np.ndarray, method: str='linear', **kwargs) -> np.ndarray:

    interp_r = np.array(pd.Series(x).interpolate(method=method,**kwargs))
    interp_l = np.array(pd.Series(interp_r[::-1]).interpolate(method=method,**kwargs))
    return interp_l[::-1]
    #return np.array(pd.Series(x).interpolate(method=method,**kwargs))


def smoothly_interpolate_nans(x: np.ndarray,
                              despike_params: dict,
                              smooth_params: dict,
                              method: str='slinear'
                             ) -> np.ndarray:
    # interpolate on despiked
    x2 = despike(x, **despike_params)
    x2 = smooth(x2, **smooth_params)
    x2 = interpolate_nans(x2, mode=method)

    # fill on original
    x3 = x.copy()
    x3[np.isnan(x)] = x2[np.isnan(x)]
    return x3



def smooth(x: np.ndarray, std: float=1.0, mode='reflect', **kwargs) -> np.ndarray:
    if np.isnan(x).any():
        return _nan_smooth(x,  std, mode=mode, **kwargs)
    else:
        return _smooth(x,  std, mode=mode, **kwargs)

def _smooth(x: np.ndarray, std: float=1.0, mode='reflect', **kwargs) -> np.ndarray:
    return gaussian_filter1d(x, std, mode=mode, **kwargs)

def _nan_smooth(x: np.ndarray, std: float=1.0, mode='reflect', **kwargs) -> np.ndarray:
    """https://stackoverflow.com/questions/18697532"""
    V = x.copy()
    V[np.isnan(x)] = 0
    VV = _smooth(V, std, mode=mode, **kwargs)

    W = 0.0 * x.copy() + 1.0
    W[np.isnan(x)] = 0.0
    WW = _smooth(W, std, mode=mode, **kwargs)

    Z = VV / (WW + 1e-10)
    Z[np.isnan(x)] = np.nan
    return Z






#@njit()
def blocking(x: np.ndarray,
             threshold: float,
             maximum_length: int,
             mean_type: str='arithmetic'
            ) -> np.ndarray:
    segments = _compute_block_segments(x, threshold, maximum_length)
    return _block_from_segments(x, tuple(segments), mean_type)

@njit()
def _compute_block_segments(x: np.ndarray,
                            threshold: float,
                            maximum_length: int
                            ) -> list:

    # find segements
    segments = [0]
    s_current = 0
    for i in range(1,x.size):
        cond1 = abs(x[i] - x[i-1]) > threshold*abs(x[i-1])
        cond2 = (i - s_current) > maximum_length
        if cond1 or cond2:
            segments.append(i)
            s_current = i

    if s_current != x.size - 1:
        segments.append(x.size - 1)

    return segments

@njit()
def _block_from_segments(x: np.ndarray,
                         segments: tuple,
                         mean_type: str='arithmetic'
                         ) -> np.ndarray:
    x_blocked = np.copy(x)
    s = segments

    for i in range(len(s)-1):
        x_seg = x[s[i]:s[i+1]]
        if mean_type == 'arithmetic':
            mean_ = np.mean(x_seg)
        elif mean_type == 'harmonic':
            mean_ = (s[i+1] - s[i]) / np.sum(1.0/x_seg)
        else:
            raise ValueError

        x_blocked[s[i]:s[i+1]] = mean_

    return x_blocked
