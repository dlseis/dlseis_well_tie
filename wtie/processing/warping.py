"""Dynamic time warping for auto strech and squeeze."""

import numpy as np

#from dtaidistance import dtw as _dtw

from wtie.processing.logs import despike, interpolate_nans, smooth
from wtie.optimize.similarity import normalized_xcorr
#from wtie.utils.types_ import List, Tuple

from scipy.interpolate import interp1d


########################
# dtaidistance
########################

# def dynamic_time_warping_lags(s1: np.ndarray,
#                               s2: np.ndarray,
#                               window: int,
#                               **kwargs) -> np.ndarray:

#     d, paths = _dtw.warping_paths(s1, s2, window=window, **kwargs)
#     best_path = _dtw.best_path(paths)

#     lags_idx = _compute_lags_from_path(best_path, s1)
#     return lags_idx


# def _compute_lags_from_path(path: List[Tuple], ref_trace: np.ndarray) -> np.ndarray:
#     """path as computed by the library dtaidistance"""
#     lags_idx = np.zeros_like(ref_trace, dtype=np.int)

#     j_pointer = 0
#     for i in range(ref_trace.size):
#         count = 0
#         value = 0.0
#         for j in range(j_pointer, len(path)):
#             point = path[j]
#             if point[0] == i:
#                 value += (point[1] - i)
#                 count += 1
#             if point[0] > i:
#                 j_pointer = j
#                 lags_idx[i] = value / count
#                 break

#     return lags_idx






########################
# xcorr
########################

def dynamic_lag(s1: np.ndarray,
                s2: np.ndarray,
                window_lenght: int,
                max_lag_idx: int
                ) -> np.ndarray:
    """Compute per-sample lag bewtween reference s1 trace and s2 trace.
    TODO: numba"""
    assert s1.size == s2.size
    lags_idx = np.zeros((s1.size,), dtype=np.int)

    assert max_lag_idx < window_lenght // 2

    # boundary
    half_length = window_lenght//2
    #pady = np.zeros((half_length,), dtype=s1.dtype)
    _std = min(np.std(s1), np.std(s2))
    pady = lambda: np.random.normal(scale=0.1*_std, size=(half_length,))
    s1 = np.concatenate((pady(), s1, pady()))
    s2 = np.concatenate((pady(), s2, pady()))


    # sliding window
    for i in range(lags_idx.size):

        # window
        j = i + half_length
        s1_w = s1[j-half_length:j+half_length]
        s2_w = s2[j-half_length:j+half_length]

        # correlation
        xcorr = normalized_xcorr(s1_w, s2_w)

        # restriction to abs max lag
        mid_xcorr_idx = xcorr.size // 2
        xcorr = xcorr[mid_xcorr_idx-max_lag_idx:mid_xcorr_idx+max_lag_idx]

        # lag
        mid_xcorr_idx = xcorr.size // 2
        lag_idx = mid_xcorr_idx - np.argmax(xcorr)
        lags_idx[i] = lag_idx

    return lags_idx


def post_process_lags_index(lags: np.ndarray,
                            median_size: int,
                            threshold: float,
                            std: float
                            ) -> np.ndarray:
    lags = np.copy(lags)

    # int to float
    lags = lags.astype(float)
    absmax = np.abs(lags).max()
    lags /= absmax

    # despke and smooth
    lags = despike(lags, median_size=median_size, threshold=threshold)
    lags = smooth(lags, std=std)
    lags = interpolate_nans(lags)



    # wrap around amplitude
    lags *= absmax
    lags = np.round(lags)

    # remove crossings
    for i in range(1, len(lags)):
        diff = lags[i] - lags[i-1] + 1
        if diff < 0:
            lags[i] -= diff

    return lags.astype(np.int)




def post_process_lags(lags: np.ndarray,
                      sampling_rate: float,
                      median_size: int,
                      threshold: float,
                      std: float
                      ) -> np.ndarray:
    lags = np.copy(lags)


    # despke and smooth
    lags = despike(lags, median_size=median_size, threshold=threshold)
    lags = smooth(lags, std=std)
    lags = interpolate_nans(lags)


    # remove crossings
    for i in range(1, len(lags)):
        diff = lags[i] - lags[i-1] + sampling_rate
        if diff < 0:
            lags[i] -= diff

    return lags


def warp(s: np.ndarray, lags_idx: np.ndarray, interpolation: str='linear',
         return_indices: bool=False
         ) -> np.ndarray:
    assert s.size == lags_idx.size

    old_indices = np.arange(s.size)
    new_indices = old_indices + lags_idx

    interp = interp1d(old_indices, s,
                      bounds_error=False, fill_value=s[-1], kind=interpolation)

    warped = interp(new_indices)
    return (warped, new_indices) if return_indices else warped