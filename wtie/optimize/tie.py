"""Utils to perform a tie."""

from time import sleep


import wtie

from wtie import grid
from wtie.optimize import logs as _logs
from wtie.optimize import similarity as _similarity
from wtie.optimize import wavelet as _wavelet


from wtie.utils.types_ import FunctionType





# For accessibility from tie
from wtie.utils.datasets.utils import InputSet, OutputSet


VERY_FINE_DT: float = 0.0005
FINE_DT: float = 0.001

def resample_seismic(seismic: grid.seismic_t, dt: float):
    """Sinc interp."""
    return grid.resample_trace(seismic, dt)

def filter_md_logs(logset: grid.LogSet, **kwargs) -> grid.LogSet:
    """Filter logs in measured depth domain.

    See function :func:`_logs.filter_logs`"""
    assert logset.is_md

    # log filtering
    filtered_logset = _logs.filter_logs(logset, **kwargs)

    return filtered_logset


def convert_logs_from_md_to_twt(logset: grid.LogSet,
                                wellpath: grid.WellPath,
                                table: grid.TimeDepthTable,
                                dt: float) -> grid.LogSet:
    """Correct the depth using the `wellpath` and converts to time
    using the time-depth `table`. Also resamples the logs in two-way-time
    at the sampling rate `dt`."""
    assert logset.is_md, "Input logs must be in measured depth."

    logset_twt = _logs.convert_logs_from_md_to_twt(logset,
                                                   table,
                                                   wellpath,
                                                   VERY_FINE_DT)
    logset_twt = grid.downsample_logset(logset_twt,
                                        new_sampling=dt
                                        )
    return logset_twt



def compute_reflectivity(logset: grid.LogSet,
                         angle_range: tuple=None
                         ) -> grid.ref_t:
    """angle_range : Tuple[start: int, end: int, delta: int],
    if provided computes angle dependent reflectivity in the
    defined range. Else computes the vertical incidence reflectivity."""
    assert logset.is_twt, "Input logs must be in two-way time."

    if angle_range is None:
        r = _logs.compute_acoustic_relfectiviy(logset)
        r.name = 'R0'
    else:
        theta_start, theta_end, delta_theta = angle_range
        r = _logs.compute_prestack_reflectivity(logset, theta_start, theta_end,
                                              delta_theta=delta_theta)
        r.name = 'Rpp'
    return r




def match_seismic_and_reflectivity(seismic: grid.seismic_t,
                                   reflectivity: grid.ref_t
                                   ):
    """Returns the `seismic` and `reflectivity` in the two-way time#
    region where they are both defined."""
    return grid.get_matching_traces(seismic, reflectivity)




def compute_wavelet(seismic: grid.seismic_t,
                    reflectivity: grid.ref_t,
                    modeler: wtie.modeling.modeling.ModelingCallable,
                    wavelet_extractor: wtie.learning.model.BaseEvaluator,
                    similarity_measure: FunctionType=None,
                    zero_phasing: bool=False,
                    scaling: bool=True,
                    scaling_params: dict=None,
                    expected_value: bool=False,
                    n_sampling: int=60
                    ) -> grid.wlt_t:

    if similarity_measure is None:
        similarity_measure = _similarity.normalized_xcorr_central_coeff

    # Compute wavelet
    if expected_value:
        if seismic.is_prestack:
            wavelet = _wavelet.compute_expected_prestack_wavelet(wavelet_extractor,
                                                                 seismic,
                                                                 reflectivity,
                                                                 zero_phasing=zero_phasing)
        else:
            wavelet = _wavelet.compute_expected_wavelet(wavelet_extractor,
                                                        seismic,
                                                        reflectivity,
                                                        zero_phasing=zero_phasing)
    else:
        if seismic.is_prestack:
            _search_wlt = _wavelet.grid_search_best_prestack_wavelet
        else:
            _search_wlt = _wavelet.grid_search_best_wavelet

        wavelet = _search_wlt(wavelet_extractor,
                              seismic,
                              reflectivity,
                              modeler,
                              similarity_measure,
                              zero_phasing=zero_phasing,
                              num_wavelets=n_sampling)

    # Find absolute scaling
    if scaling:
        _scale_wlt = _wavelet.scale_prestack_wavelet if seismic.is_prestack \
            else _wavelet.scale_wavelet

        print("Find wavelet absolute scale")
        sleep(1.0)
        wavelet, _ = _scale_wlt(wavelet,
                                seismic,
                                reflectivity,
                                modeler,
                                min_scale=scaling_params['wavelet_min_scale'],
                                max_scale=scaling_params['wavelet_max_scale'],
                                num_iters=scaling_params.get('num_iters', 70)
                                )
    return wavelet




def compute_synthetic_seismic(modeler: wtie.modeling.modeling.ModelingCallable,
                              wavelet: grid.wlt_t,
                              reflectivity: grid.ref_t
                              ) -> grid.seismic_t:
    if type(wavelet) is grid.PreStackWavelet:
        return _wavelet.compute_synthetic_prestack_seismic(modeler, wavelet, reflectivity)
    elif type(wavelet) is grid.Wavelet:
            return _wavelet.compute_synthetic_seismic(modeler, wavelet, reflectivity)
    else:
        raise TypeError