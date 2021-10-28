"""Utils for similarity measures"""

import numpy as np

from wtie.processing import grid


def pep(seismic: grid.Seismic,
        synthetic: grid.Seismic,
        normalize: bool=False) -> float:
    """https://www.researchgate.net/publication/287551452_Tutorial_Good_practice_in_well_ties"""

    if normalize:
        seismic_values = seismic.values / np.abs(seismic.values).max()
        synth_values = synthetic.values / np.abs(synthetic.values).max()
    else:
        seismic_values = seismic.values
        synth_values = synthetic.values

    trace_energy = energy(seismic_values)

    residual = seismic_values - synth_values
    residual_energy = energy(residual)

    p = 1.0 - residual_energy / trace_energy

    return p


def normalized_xcorr_maximum(a: np.ndarray, b: np.ndarray) -> float:
    return normalized_xcorr(a, b).max()

def normalized_xcorr_central_coeff(a: np.ndarray, b: np.ndarray) -> float:
    xcorr = normalized_xcorr(a, b)
    return xcorr[xcorr.size//2]



def normalized_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - np.mean(a)) / (np.std(a))
    b = (b - np.mean(b)) / (np.std(b))
    xcorr = np.correlate(a, b, 'full') / max(len(a), len(b))
    return xcorr


def energy(x: np.ndarray) -> float:
    return np.sum(np.square(x))



def traces_normalized_xcorr(trace1: grid.BaseTrace,
                            trace2: grid.BaseTrace
                            ) -> grid.XCorr:

    # verify basis
    assert trace1.basis_type == trace2.basis_type
    assert np.allclose(trace1.sampling_rate, trace2.sampling_rate)

    # xcorr
    xcorr = normalized_xcorr(trace1.values, trace2.values)

    # lag
    sr = trace1.sampling_rate
    duration = xcorr.size * sr
    lag = np.arange(-duration/2, (duration-sr)/2, sr)

    if trace1.is_twt:
        btype = 'tlag'
    else:
        btype = 'zlag'

    return grid.XCorr(xcorr, lag, btype, name='XCorr')



def prestack_traces_normalized_xcorr(trace1: grid.BasePrestackTrace,
                                     trace2: grid.BasePrestackTrace
                                     ) -> grid.PreStackXCorr:

    assert (trace1.angles == trace2.angles).all()

    xcorr = []
    for theta in trace1.angles:
        xc = traces_normalized_xcorr(trace1[theta], trace2[theta])
        xc.theta = theta
        xcorr.append(xc)

    return grid.PreStackXCorr(xcorr)


def prestack_mean_central_xcorr_coeff(trace1: grid.BasePrestackTrace,
                                      trace2: grid.BasePrestackTrace
                                      ) -> float:
    xcorr = prestack_traces_normalized_xcorr(trace1, trace2)
    return np.mean(xcorr.Rc)

def central_xcorr_coeff(trace1: grid.trace_t,
                        trace2: grid.trace_t
                        ) -> float:
    if trace1.is_prestack:
        return prestack_mean_central_xcorr_coeff(trace1, trace2)
    else:
        return normalized_xcorr_central_coeff(trace1.values, trace2.values)


def dynamic_normalized_xcorr(trace1: np.ndarray,
                             trace2: np.ndarray,
                             window_lenght: float=0.070
                             ) -> grid.DynamicXCorr:

    assert np.allclose(trace1.basis, trace2.basis, atol=1e-3)

    half_index = int(round(window_lenght / trace1.sampling_rate)) // 2


    # boundary
    _std = min(np.std(trace1.values), np.std(trace2.values))
    pady = lambda: np.random.normal(scale=0.1*_std, size=(half_index,))
    s1 = np.concatenate((pady(), trace1.values, pady()))
    s2 = np.concatenate((pady(), trace2.values, pady()))

    # prealloc
    s1_w_ = s1[:2*half_index]
    s2_w_ = s2[:2*half_index]
    xcorr_ = normalized_xcorr(s1_w_, s2_w_)
    dxcorr = np.zeros((trace1.basis.size, xcorr_.size))


    # sliding window
    for i in range(trace1.size):
        # window
        j = i + half_index
        s1_w = s1[j-half_index:j+half_index]
        s2_w = s2[j-half_index:j+half_index]

        # correlation
        dxcorr[i,:] = normalized_xcorr(s1_w, s2_w)


    return grid.DynamicXCorr(dxcorr, trace1.basis,
                             grid._inverted_name(trace1.basis_type),
                             name="Dynamic X-Correlation")
