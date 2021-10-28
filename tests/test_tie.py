import pytest

import numpy as np

from wtie import grid
from wtie import tie
from wtie.optimize import similarity
from wtie.optimize import logs
from wtie.optimize import wavelet


from wtie.learning.network import VariationalNetwork
from wtie.learning.model import VariationalEvaluator
from wtie.modeling.modeling import ConvModeler



@pytest.fixture
def input_data():
    # Logs
    basis = np.arange(1200, 2500, step=1.5)
    basis_type = 'md'
    vp = grid.Log(np.random.uniform(2800, 4060, size=basis.shape), basis, basis_type)
    vs = grid.Log(0.7*vp.values, basis, basis_type)
    rho = grid.Log(np.random.uniform(1.2, 5.6, size=basis.shape), basis, basis_type)
    logset_md = grid.LogSet({'Vp':vp, 'Rho':rho, 'Vs':vs})

    # Well path
    wp = grid.WellPath(md=np.arange(0, 3000, step=1.5), kb=25) # vertical well

    # TD table
    vp_tvd = grid._convert_log_from_md_to_tvdss(vp,wp)
    twt = 1.2 + 0.004*np.arange(len(vp_tvd))
    tdt = grid.TimeDepthTable(twt, vp_tvd.basis)

    # Seismic
    angles = np.arange(0,30,10)
    #seismic = grid.Seismic(np.random.normal(size=(twt.size,)), twt, 'twt')
    seismic = grid.PreStackSeismic(tuple([\
        grid.Seismic(np.random.normal(size=(twt.size,)), twt, 'twt', theta=i) for i in angles])
                                   )

    # Inputs
    inputs = tie.InputSet(logset_md, seismic, wp, tdt)

    # Evaluator
    _net = VariationalNetwork(112)
    wavelet_extractor = VariationalEvaluator(network=_net, expected_sampling=0.002)

    # Modeler
    modeler = ConvModeler()



    return inputs, wavelet_extractor, modeler


def test_tie_v1(input_data, prestack: bool=False):

    # input
    inputs, wavelet_extractor, modeler = input_data

    # stack / prestack
    if prestack:
        seismic = inputs.seismic
    else:
        seismic = inputs.seismic.traces[0]


    # filtering
    filtered_logset_md = tie.filter_md_logs(inputs.logset_md,
                                        median_size=21, threshold=2.0,
                                        std=2.0, std2=None)

    # conversion
    logset_twt = tie.convert_logs_from_md_to_twt(filtered_logset_md,
                                                 inputs.wellpath,
                                                 inputs.table,
                                                 wavelet_extractor.expected_sampling)


    # reflectivity
    r = tie.compute_reflectivity(logset_twt, seismic.angle_range)


    # interp seismic
    seismic_sinc = tie.resample_seismic(seismic, wavelet_extractor.expected_sampling)


    # matching
    seis_match, r_match = tie.match_seismic_and_reflectivity(seismic_sinc, r)

    # wavelet extraction
    if prestack:
        pred_wlt = wavelet.compute_expected_prestack_wavelet(\
                                                    evaluator=wavelet_extractor,
                                                    seismic=seis_match,
                                                    reflectivity=r_match)
    else:
        pred_wlt = wavelet.compute_expected_wavelet(evaluator=wavelet_extractor,
                                                seismic=seis_match,
                                                reflectivity=r_match)



    # modeling
    synth_seismic = tie.compute_synthetic_seismic(modeler, pred_wlt, r_match)

    # fit
    if prestack:
        xcorr = similarity.prestack_traces_normalized_xcorr(seis_match, synth_seismic)
    else:
        xcorr = similarity.traces_normalized_xcorr(seis_match, synth_seismic)



def test_prestack_tie_v1(input_data):
    return test_tie_v1(input_data, prestack=True)










