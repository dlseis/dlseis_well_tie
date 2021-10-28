import numpy as np

from wtie.processing import reflection
from wtie.modeling.wavelet import ricker
from wtie.optimize.logs import (compute_acoustic_relfectiviy,
                                compute_synthetic_seismic,
                                convert_logs_from_md_to_twt)

from wtie.processing import grid


def test_reflectivity_computing():
    N = 51
    vp = np.random.uniform(low=2000, high=6500, size=(N,))
    vs = 0.6*vp
    rho = np.random.uniform(low=1.1, high=5.3, size=(N,))

    R0 = reflection.vertical_acoustic_reflectivity(vp, rho)
    rpp_theta_0 = reflection.zoeppritz_rpp(vp, vs, rho, theta=0)
    rpp = reflection.prestack_rpp(vp, vs, rho, 0, 15, delta_theta=2)

    assert np.allclose(R0, rpp_theta_0)
    assert np.allclose(R0, rpp[0,...])



def test_transformations():
    tdt = grid.TimeDepthTable([0.0, 0.1, 0.3, 0.5, 1.2],
                              [0.0, 154, 198, 489, 2292])



    N = 101
    dz = 5
    dt = 0.004
    _z_range = 122 + np.arange(N)*dz
    _vp = np.random.uniform(low=2000, high=6500, size=(N,))
    _rho = np.random.uniform(low=1.1, high=5.3, size=(N,))

    _md = np.arange(3*N)*dz
    wellpath = grid.WellPath(md=_md)
    _inclination = np.random.uniform(-30,30,size=(_md.size - 1,))
    _tvd = grid.WellPath.get_tvdkb_from_inclination(_md, _inclination)
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, 25)
    wellpath = grid.WellPath(md=_md, tvdss=_tvd, kb=25)

    logs_md = grid.LogSet({'Vp':grid.Log(_vp, _z_range, 'md'),
                          'Rho':grid.Log(_rho, _z_range, 'md')})

    logs_twt = convert_logs_from_md_to_twt(logs_md, tdt, wellpath, dt)

    assert np.allclose(logs_md.AI.basis, logs_md.Vp.basis)

    r0 = compute_acoustic_relfectiviy(logs_twt)


    _t, _wlt = ricker(f=28, dt=dt, n_samples=52)
    wavelet = grid.Wavelet(_wlt, _t)

    synth_seismic = compute_synthetic_seismic(wavelet, r0)

    assert np.allclose(r0.basis, synth_seismic.basis)

    synth_seis_interp = grid.upsample_trace(synth_seismic, dt/2)
    synth_seis_interp = grid.downsample_trace(synth_seis_interp, dt)
