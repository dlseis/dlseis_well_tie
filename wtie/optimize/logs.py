"""Search on optimal logs calibration."""
import random
import numpy as np



from wtie.processing import grid
#from wtie.processing.grid import convert_log_from_md_to_twt as _md_to_twt
#from wtie.processing.logs import despike, interpolate_nans, smooth, blocking
#from wtie.processing.logs import _compute_block_segments, _block_from_segments
from wtie.processing import logs as _logs
#from wtie.processing.reflection import vertical_acoustic_reflectivity, prestack_rpp
from wtie.processing import reflection as _reflection
from wtie.modeling.modeling import convolution_modeling
from wtie.utils.types_ import List


##################
# LOGS
##################
def update_log_values(new_values: np.ndarray, current_log: grid.Log) -> grid.Log:
    return grid.update_trace_values(new_values, current_log)
    #return grid.Log(new_values,
    #                current_log.basis,
    #                grid._inverted_name(current_log.basis_type),
    #                name=current_log.name)


def old_temporal_strech_squeeze(table: grid.TimeDepthTable,
                            central_idx: int,
                            delta_idx: int,
                            pert_ratio: float,
                            dt : float=0.012,
                            mode: str='slinear'
                            ) -> grid.TimeDepthTable:

    table = table.temporal_interpolation(dt=dt)
    assert central_idx - delta_idx > 0
    assert central_idx + delta_idx < table.size


    twt_ = np.concatenate((table.twt[:central_idx-delta_idx],
                           table.twt[central_idx]*(1+pert_ratio)*np.ones((1,)),
                           table.twt[central_idx+delta_idx:])
                         )
    tvd_ = np.concatenate((table.tvdss[:central_idx-delta_idx],
                           table.tvdss[central_idx]*np.ones((1,)),
                           table.tvdss[central_idx+delta_idx:])
                         )
    tdt_ = grid.TimeDepthTable(twt_, tvd_)
    return tdt_.temporal_interpolation(dt=dt, mode=mode)


def filter_log(log: grid.Log,
               median_size: int=31,
               threshold: float=4.0,
               std: float=2.0,
               std2: float=None,
              ) -> grid.Log:
    f_log = _logs.despike(log.values, median_size=median_size, threshold=threshold)
    f_log = _logs.interpolate_nans(f_log)
    f_log = _logs.smooth(f_log, std=std)
    if std2:
        f_log = _logs.smooth(f_log, std=std2)

    return update_log_values(f_log, log)


def filter_logs(logset: grid.LogSet,
                median_size: int=31,
                threshold: float=4.0,
                std: float=2.0,
                std2: float=None,
                log_keys: List[str] = None
               ) -> grid.LogSet:
    """Apply processing to all logs in the logset, expect if log_keys filter
    is specified, then only applies to those logs.
    """

    keys_ = logset.Logs.keys() if log_keys is None else log_keys

    new_logs = {}
    for key_ in keys_:
        new_logs[key_] = filter_log(logset[key_], median_size, threshold, std, std2)

    return grid.LogSet(new_logs)




def block_logs(logset: grid.LogSet,
               threshold_perc: float,
               maximum_length: int=None,
               baseline: str = 'AI',
               log_keys: List[str] = None
               ) -> grid.LogSet:
    """Apply processing to all logs in the logset, expect if log_keys filter
    is specified, then only applies to those logs.
    """
    # Filter log keys
    keys_ = logset.Logs.keys() if log_keys is None else log_keys

    # Parameters
    threshold = threshold_perc / 100

    if maximum_length is None:
        maximum_length = int(round(logset.basis.size // 4))

    # Compute segments
    if baseline == 'Vp':
        segments = _logs._compute_block_segments(logset.vp, threshold, maximum_length)
        segments = len(keys_) * [segments]
    elif baseline == 'AI':
        segments = _logs._compute_block_segments(logset.ai, threshold, maximum_length)
        segments = len(keys_) * [segments]
    elif baseline == 'itself':
        segments = [_logs._compute_block_segments(logset.Logs[key_].values, threshold, maximum_length) \
                    for key_ in keys_]
    else:
        raise ValueError
    segments = [tuple(s) for s in segments]

    # Block logs
    new_logs = {}
    for i, key_ in enumerate(keys_):
        log = logset[key_]
        if key_ in ['Vp', 'Vs']:
            log_b = _logs._block_from_segments(log.values, segments[i], 'harmonic')
        else:
            log_b = _logs._block_from_segments(log.values, segments[i], 'arithmetic')


        new_logs[key_] = update_log_values(log_b, log)

    return grid.LogSet(new_logs)


###################
# TD TABLES
###################

_apply_poly = lambda x, p : np.poly1d(p)(x)

def _perturbe_poly(p, delta_abs):
    return [p_i + p_i*(random.uniform(-delta_abs,delta_abs)) \
            for i, p_i in enumerate(p[::-1])][::-1]

def OLDget_perturbed_time_depth_tables(tdt: grid.TimeDepthTable,
                                    n: int=100,
                                    delta_abs: float=0.03,
                                    order: int=5
                                    ) -> List[grid.TimeDepthTable]:

    tables = []
    poly = np.polyfit(tdt.tvdss[1:], tdt.twt[1:], order)

    i = 0
    _i = 0
    while i < n:
        _i += 1
        poly_pert = _perturbe_poly(poly,delta_abs)
        p_twt_ = _apply_poly(tdt.tvdss[1:], poly_pert)
        p_twt_ = np.concatenate((np.zeros(1,), p_twt_))
        try:
            tables.append(grid.TimeDepthTable(twt=p_twt_, tvdss=tdt.tvdss))
            i += 1
        except:
            # unrealsitic perurbations
            if _i > 4*n: raise ValueError
            continue

    return tables



def get_tdt_from_vp(Vp: grid.Log,
                    tdt: grid.TimeDepthTable,
                    wp: grid.WellPath = None
                    ) -> grid.TimeDepthTable:

    if Vp.is_md:
        t_start, z_error = grid.TimeDepthTable.get_twt_start_from_checkshots(Vp, wp, tdt)
        sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp,
                                                                        wp=wp,
                                                                        origin=t_start)
    elif Vp.is_twt:
        z_start, t_error = grid.TimeDepthTable.get_tvdss_start_from_checkshots(Vp, tdt)
        sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp,
                                                                        origin=z_start)

    else:
        raise NotImplementedError()

    return sonic_tdt_pert



def OLD_get_pertubed_tdt_from_vp(Vp: grid.Log,
                             wp: grid.WellPath,
                             tdt: grid.TimeDepthTable,
                             p_pert_ratio: float=0.02,
                             t_pert_ratio: float=0.01,
                             max_degree: int=5,
                             N: int=50
                            ) -> List[grid.TimeDepthTable]:
    assert Vp.is_md

    tables = []

    i = 0
    i_ = 0
    while i < N:
        i_ += 1
        deg = np.random.randint(1,max_degree+1)
        poly = np.polyfit(Vp.basis, Vp.values, deg)
        poly_pert = _perturbe_poly(poly, p_pert_ratio**deg)

        vl = _apply_poly(Vp.basis, poly)
        vl_p = _apply_poly(Vp.basis, poly_pert)
        Vp_p = grid.Log(Vp.values - vl + vl_p, Vp.basis, 'md')

        t_start, z_error = grid.TimeDepthTable.get_tvdss_start_from_checkshots(Vp_p, wp, tdt)
        t_start += (random.uniform(-t_pert_ratio, t_pert_ratio) * t_start)

        try:
            sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp_p, wp, t_start=t_start)
            tables.append(sonic_tdt_pert)
            i += 1
        except:
            # unrealsitic perurbations
            if i_ > 4*N: raise ValueError
            continue

    return tables

#####################
# OTHER
#####################



def compute_prestack_reflectivity(logs: grid.LogSet,
                                  theta_start: int,
                                  theta_end: int,
                                  delta_theta: int=2
                                  ) -> grid.PreStackReflectivity:
    """reflectivity from Vp, Vs and rho.
    Basis must be twt in seconds.
    """

    # verify basis
    assert logs.is_twt

    values = _reflection.prestack_rpp(logs.vp, logs.vs, logs.rho, theta_start,
                                theta_end, delta_theta)

    thetas = range(theta_start, theta_end+delta_theta, delta_theta)

    reflectivities = [grid.Reflectivity(values[i,:], logs.basis[1:], theta=thetas[i]) \
                      for i in range(values.shape[0])]

    return grid.PreStackReflectivity(reflectivities)


def compute_acoustic_relfectiviy(logs: grid.LogSet) -> grid.Reflectivity:
    """Vertical incidence reflectivity from Vp and rho.
    Basis must be twt in seconds.
    """

    # verify basis
    assert logs.is_twt

    # R0
    reflectivity = _reflection.vertical_acoustic_reflectivity(logs.vp,
                                                  logs.rho)

    return grid.Reflectivity(reflectivity, logs.basis[1:])




def convert_logs_from_md_to_tvdss(logset: grid.LogSet,
                                  trajectory: grid.WellPath,
                                  dz: float=None,
                                  interpolation='linear'
                                  ) -> grid.LogSet:
    # verify basis
    assert logset.is_md

    logs_tvd = grid._apply_trace_process_to_logset(\
                            grid._convert_log_from_md_to_tvdss,
                            logset,
                            trajectory,
                            dz=dz,
                            interpolation=interpolation)
    return logs_tvd



def convert_logs_from_md_to_twt(logset: grid.LogSet,
                                table: grid.TimeDepthTable,
                                trajectory: grid.WellPath,
                                dt: float,
                                interpolation='linear'
                                ) -> grid.LogSet:
    # verify basis
    assert logset.is_md

    logs_twt = grid._apply_trace_process_to_logset(\
                            grid.convert_log_from_md_to_twt,
                            logset,
                            table,
                            trajectory,
                            dt,
                            interpolation=interpolation)
    return logs_twt


def compute_synthetic_seismic(wavelet: grid.Wavelet,
                              reflectivity: grid.Reflectivity
                              ) -> grid.Seismic:
    # verify basis
    assert reflectivity.is_twt
    assert np.allclose(reflectivity.sampling_rate, wavelet.sampling_rate,
                       atol=1e-5) # tolerance at 0.1 millisecond

    seismic = convolution_modeling(wavelet.values,
                                   reflectivity.values,
                                   noise=None,
                                   mode='same')
    return grid.Seismic(seismic, reflectivity.basis, 'twt')



def compute_synthetic_prestack_seismic(wavelet: grid.PreStackWavelet,
                                       reflectivity: grid.PreStackReflectivity
                                       ) -> grid.PreStackSeismic:
    # verify basis
    assert reflectivity.is_twt
    assert np.allclose(reflectivity.sampling_rate, wavelet.sampling_rate,
                       atol=1e-5) # tolerance at 0.1 millisecond

    assert wavelet.angles == reflectivity.angles

    seismic = []
    for theta in wavelet.angles:
        values = convolution_modeling(wavelet[theta].values,
                                      reflectivity[theta].values,
                                      noise=None,
                                      mode='same')

        seismic.append(grid.Seismic(values, reflectivity.basis, 'twt',
                                    theta=theta))

    return grid.PreStackSeismic(seismic)



