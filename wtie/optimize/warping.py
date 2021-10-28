"""Dynamic time warping for auto strech and squeeze."""
import numpy as np

from scipy.interpolate import interp1d

from wtie.processing import grid
from wtie.processing import warping as _warping
from wtie.optimize import logs as _logs
from wtie.processing.logs import interpolate_nans as _interpolate_nans
from wtie.utils.types_ import List, Tuple



def NOTUSEDcompute_dynamic_time_warping_lag(ref_trace: grid.BaseTrace,
                        other_trace: grid.BaseTrace,
                        max_lag: float,
                        post_process: dict=None,
                        dtw_kwargs: dict=None) -> grid.DynamicLag:
    """Compute dynamic time warping lags between reference and other trace.
    Units of window_length and max_lag must be the same as unit of
    ref_trace.sampling_rate.

    computed with dtaidistance

    ref_trace: real seismic
    other_trace: synthetic seismic

    post process example:
    post_process = dict(median_size=11,threshold=0.5, std=0.5)
    """


    assert ref_trace.basis_type == other_trace.basis_type
    assert np.allclose(ref_trace.basis, other_trace.basis, atol=1e-3)

    max_lag_idx = int(round(max_lag / ref_trace.sampling_rate))

    if dtw_kwargs is None:
        dtw_kwargs = {}

    lags_idx = _warping.dynamic_time_warping_lags(ref_trace.values, other_trace.values,
                            window=max_lag_idx, **dtw_kwargs)



    lags = lags_idx.astype(ref_trace.basis.dtype) * ref_trace.sampling_rate

    if post_process is not None:
        raise NotImplementedError("Need to find a postprocessing that \
                                  does not lead to wrong stretch & squeeze.")
        lags = _warping.post_process_lags(lags, ref_trace.sampling_rate,
                                          **post_process)



    return grid.DynamicLag(lags, ref_trace.basis,
                           grid._inverted_name(ref_trace.basis_type))



def NOTUSED_compute_lags_from_path(path: List[Tuple], ref_trace: grid.BaseTrace) -> grid.DynamicLag:
    """path as computed by the library dtaidistance"""
    lags_idx = np.zeros_like(ref_trace.basis)

    j_pointer = 0
    for i in range(ref_trace.size):
        count = 0
        value = 0.0
        for j in range(j_pointer, len(path)):
            point = path[j]
            if point[0] == i:
                value += (point[1] - i)
                count += 1
            if point[0] > i:
                j_pointer = j
                lags_idx[i] = value / count
                break


    lags = ref_trace.sampling_rate * lags_idx

    return grid.DynamicLag(lags, ref_trace.basis, grid._inverted_name(ref_trace.basis_type))




def _dirty_remove_deacreasing_values(twt: np.ndarray, tvd: np.ndarray):
    #TODO: numba
    valid_twt = [twt[0]]
    valid_tvd = [tvd[0]]

    current_twt = twt[0]
    current_tvd = tvd[0]
    for i in range(1, len(twt)):
        if (twt[i] > current_twt) and (tvd[i] > current_tvd):
            valid_twt.append(twt[i])
            valid_tvd.append(tvd[i])
            current_twt = twt[i]
            current_tvd = tvd[i]

    valid_twt = np.array(valid_twt)
    valid_tvd = np.array(valid_tvd)

    assert ((valid_twt[1:] - valid_twt[:-1]) >= 0).all()
    assert ((valid_tvd[1:] - valid_tvd[:-1]) >= 0).all()

    return valid_twt, valid_tvd




def apply_lags_to_table(table: grid.TimeDepthTable,
                        lags: grid.DynamicLag,
                        post_process: dict=None
                        ) -> grid.TimeDepthTable:
    """TODO: CHECK """
    assert lags.is_twt

    # consatnt sampling
    table = table.temporal_interpolation(dt=lags.sampling_rate)

    # init
    start_idx = np.argmin(np.abs(table.twt - lags.basis[0]))

    # stretch & squeeze
    ss_twt = np.copy(table.twt)
    for i in range(len(lags)):
        #ss_twt[start_idx + i] += lags.values[i] # TODO CHECK
        ss_twt[start_idx + i] -= lags.values[i]

    #interp = interp1d(interp_table.twt, interp_table.tvdss,
    #              bounds_error=False, fill_value=interp_table.tvdss[-1], kind='linear')

    interp = interp1d(ss_twt, table.tvdss,
                  bounds_error=False, fill_value="extrapolate", kind='linear')
    ss_tvdss = interp(table.twt) # rose mary

    # remove decreasing values
    newer_twt = table.twt
    newer_tvdss = ss_tvdss

    newer_twt, newer_tvdss = _dirty_remove_deacreasing_values(newer_twt, newer_tvdss)

    # inf?
    if (newer_twt[-1] == np.inf) or (newer_tvdss[-1] == np.inf):
        newer_twt = newer_twt[:-1]
        newer_tvdss = newer_tvdss[:-1]


    ss_table = grid.TimeDepthTable(twt=newer_twt, tvdss=newer_tvdss)

    if post_process is not None:
        raise NotImplementedError("Need to find a postprocessing that \
                                  does not lead to wrong stretch & squeeze.")
        ss_table = _filter_table(ss_table, post_process)

    return ss_table


def _filter_table(table: grid.TimeDepthTable,
                  filter_params: dict,
                  ) -> grid.TimeDepthTable:
    # ASSUMES Vp in m/s
    slope_twt = np.copy(table.slope_velocity_twt().values)

    # despike and smooth


    # clip
    slope_twt[slope_twt > filter_params['max_velocity']] = filter_params['max_velocity']
    slope_twt[slope_twt < filter_params['min_velocity']] = filter_params['min_velocity']


    # np to trace
    slope_twt = grid.update_trace_values(slope_twt, table.slope_velocity_twt())

    return _logs.get_tdt_from_vp(slope_twt, table)




def compute_dynamic_lag(ref_trace: grid.BaseTrace,
                        other_trace: grid.BaseTrace,
                        window_length: float,
                        max_lag: float,
                        post_process: dict=None) -> grid.DynamicLag:
    """Compute dynamic time warping lags between reference and other trace.
    Units of window_length and max_lag must be the same as unit of
    ref_trace.sampling_rate (seconds).

    ref_trace: real seismic
    other_trace: synthetic seismic
    window_length: length of the sliding cross-correaltion window. Precision vs
    stability trade-off.
    max_lag: maximum lag of the cross-correlation.

    post process example:
    post_process = dict(median_size=11,threshold=0.5, std=0.5)
    """


    assert ref_trace.basis_type == other_trace.basis_type
    assert np.allclose(ref_trace.basis, other_trace.basis, atol=1e-3)

    window_length_idx = int(round(window_length / ref_trace.sampling_rate))
    max_lag_idx = int(round(max_lag / ref_trace.sampling_rate))

    lags_idx = _warping.dynamic_lag(ref_trace.values, other_trace.values,
                            window_length_idx, max_lag_idx)

    # if post_process is not None:
    #     lags_idx = _warping.post_process_lags_index(lags_idx, **post_process)


    lags = lags_idx.astype(ref_trace.basis.dtype) * ref_trace.sampling_rate

    if post_process is not None:
        raise NotImplementedError("Need to find a postprocessing that \
                                  does not lead to wrong stretch & squeeze.")
        lags = _warping.post_process_lags(lags, ref_trace.sampling_rate,
                                          **post_process)



    return grid.DynamicLag(lags, ref_trace.basis,
                           grid._inverted_name(ref_trace.basis_type))



def warp_trace(trace: grid.BaseTrace, lags: grid.DynamicLag,
               interpolation: str='linear') -> grid.BaseTrace:
    assert trace.basis_type == lags.basis_type
    assert np.allclose(trace.basis, lags.basis, atol=1e-3)

    lags_idx = np.round(lags.values / trace.sampling_rate).astype(np.int)
    new_values, new_indices = _warping.warp(trace.values, lags_idx, return_indices=True)
    #new_basis = ... #???
    return grid.update_trace_values(new_values, trace) # assumes same basis


