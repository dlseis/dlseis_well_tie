"""Some functions to perform auto well-tie."""
from tqdm import tqdm
from time import sleep

import wtie

from wtie.processing import grid
from wtie.utils.datasets.utils import InputSet, OutputSet

from wtie.optimize import similarity as _similarity
from wtie.optimize import warping as _warping
from wtie.optimize import optimizer as _optimizer
from wtie.optimize import tie as _tie


# Some constants affecting the workflow
# when both values are True, results are a bit less good but final wavelets
# have smaller absolute phase.
# when both values are False, results are better, but wavelets can have strong
# positive or negative phase.
# when INTERMEDIATE_EXPECTED_VALUE is False, prestack auto-tie can take twice as long.
INTERMEDIATE_ZERO_PHASING: bool = True
INTERMEDIATE_EXPECTED_VALUE: bool = True


def stretch_and_squeeze(inputs: InputSet,
                        current_outputs: OutputSet,
                        wavelet_extractor: wtie.learning.model.BaseEvaluator,
                        modeler: wtie.modeling.modeling.ModelingCallable,
                        wavelet_scaling_params: dict,
                        best_params: dict,
                        stretch_and_squeeze_params: dict
                        ):

    from_seismic = current_outputs.seismic
    to_seismic = current_outputs.synth_seismic

    if inputs.seismic.is_prestack:
        first_angle = from_seismic.angles[0]
        ref_angle = stretch_and_squeeze_params.get(
            'reference_angle', first_angle)
        stretch_and_squeeze_params.pop('reference_angle', None)
        from_seismic = from_seismic[ref_angle]
        to_seismic = to_seismic[ref_angle]

    dlags = _warping.compute_dynamic_lag(from_seismic,
                                         to_seismic,
                                         **stretch_and_squeeze_params)

    warped_table = _warping.apply_lags_to_table(current_outputs.table, dlags)

    outputs = _intermediate_tie_v1(inputs.logset_md,
                                   inputs.wellpath,
                                   warped_table,
                                   inputs.seismic,
                                   wavelet_extractor,
                                   modeler,
                                   best_params)

    outputs.dlags = dlags

    # Final wavelet
    wavelet = _tie.compute_wavelet(outputs.seismic,
                                   outputs.r,
                                   modeler,
                                   wavelet_extractor,
                                   zero_phasing=False,
                                   scaling=True,
                                   expected_value=False,
                                   scaling_params=wavelet_scaling_params)
    # final synth
    synth_seismic = _tie.compute_synthetic_seismic(modeler, wavelet, outputs.r)

    # overwrite w/ new data
    outputs.wavelet = wavelet
    outputs.synth_seismic = synth_seismic

    # similarity
    if not inputs.seismic.is_prestack:
        xcorr = _similarity.traces_normalized_xcorr(outputs.seismic,
                                                    outputs.synth_seismic)
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = _similarity.dynamic_normalized_xcorr(outputs.seismic,
                                                      outputs.synth_seismic)
    else:
        xcorr = _similarity.prestack_traces_normalized_xcorr(outputs.seismic,
                                                             outputs.synth_seismic)
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = None

    outputs.xcorr = xcorr
    outputs.dxcorr = dxcorr

    return outputs


def tie_v1(inputs: InputSet,
           wavelet_extractor: wtie.learning.model.BaseEvaluator,
           modeler: wtie.modeling.modeling.ModelingCallable,
           wavelet_scaling_params: dict,
           search_params: dict = None,
           search_space: dict = None,
           stretch_and_squeeze_params: dict = None,
           ) -> OutputSet:
    """
    Utility to perform automatic (prestack) seismic to well tie. This version 1
    serves as a base recipe. Feel free to implement your own recipe using the
    various tools of the package.

    Parameters
    ----------
    inputs : _tie.InputSet
        Necessary inputs for the well tie.
    wavelet_extractor : wtie.learning.model.BaseEvaluator
        Object to extract a wavelet using the provided neural network.
    modeler : wtie.modeling.modeling.ModelingCallable
        Synthetic modeling tool.
    wavelet_scaling_params : dict
        Parameters for the search of the optimal absolute wavelet sacle.
        3 parameters: 'wavelet_min_scale' and 'wavelet_max_scale' are the search
        bounds. 'num_iters' (optional) is the total number of iterations for the
        search.
    search_space : dict, optional
        Bounds of the search space. See `get_default_search_space_v1`.
    search_params : dict, optional
        Dict with 3 parameters for the Bayesian search. 'num_iters' is the total
        number of iteration. 'similarity_std' is an estimation of the uncertainty
        of the similarity measure to be maximized. 'random_ratio' is the ratio of
        random search vs bayesian search. A value of .6 means that 60% of iterations
        will correspond to random search.
    stretch_and_squeeze_params : dict, optional
        Parameters for the optinal strecth and squeeze. 2 parameters are 'window_length'
        and 'max_lag' both in seconds. See `_warping.compute_dynamic_lag`.

    Returns
    -------
    outputs : _tie.OutputSet
    """

    if search_space is None:
        search_space = get_default_search_space_v1()

    # Search
    # Optimize for best parameters
    # at this point, the wavelet is zero-phased
    if search_params is None:
        search_params = {}
    num_iters = search_params.get('num_iters', 80)
    similarity_std = search_params.get('similarity_std', 0.01)
    random_ratio = search_params.get('random_ratio', 0.6)

    ax_client = _search_best_params_v1(inputs,
                                       wavelet_extractor,
                                       modeler,
                                       search_space,
                                       num_iters,
                                       random_ratio,
                                       similarity_std)
    best_params = ax_client.get_best_parameters()[0]

    # Intermediate tie
    shifted_table = grid.TimeDepthTable.t_bulk_shift(inputs.table,
                                                     best_params['table_t_shift']
                                                     )

    outputs_tmp1 = _intermediate_tie_v1(inputs.logset_md,
                                        inputs.wellpath,
                                        shifted_table,
                                        inputs.seismic,
                                        wavelet_extractor,
                                        modeler,
                                        best_params)

    # Optional stretch and squeeze
    if stretch_and_squeeze_params is not None:
        from_seismic = outputs_tmp1.seismic
        to_seismic = outputs_tmp1.synth_seismic
        if outputs_tmp1.seismic.is_prestack:
            first_angle = from_seismic.angles[0]
            ref_angle = stretch_and_squeeze_params.get(
                'reference_angle', first_angle)
            stretch_and_squeeze_params.pop('reference_angle', None)
            from_seismic = from_seismic[ref_angle]
            to_seismic = to_seismic[ref_angle]

        dlags = _warping.compute_dynamic_lag(from_seismic,
                                             to_seismic,
                                             **stretch_and_squeeze_params)

        warped_table = _warping.apply_lags_to_table(outputs_tmp1.table, dlags)

        outputs_tmp2 = _intermediate_tie_v1(inputs.logset_md,
                                            inputs.wellpath,
                                            warped_table,
                                            inputs.seismic,
                                            wavelet_extractor,
                                            modeler,
                                            best_params)

        outputs_tmp2.dlags = dlags

    else:
        outputs_tmp2 = outputs_tmp1

    # Final wavelet
    wavelet = _tie.compute_wavelet(outputs_tmp2.seismic,
                                   outputs_tmp2.r,
                                   modeler,
                                   wavelet_extractor,
                                   zero_phasing=False,
                                   scaling=True,
                                   expected_value=False,
                                   scaling_params=wavelet_scaling_params)

    # Final synthetic
    synth_seismic = _tie.compute_synthetic_seismic(
        modeler, wavelet, outputs_tmp2.r)

    # overwrite w/ new data
    outputs_tmp2.ax_client = ax_client
    outputs_tmp2.wavelet = wavelet
    outputs_tmp2.synth_seismic = synth_seismic

    # Similarity between synthetic and real seismic
    if not inputs.seismic.is_prestack:
        xcorr = _similarity.traces_normalized_xcorr(outputs_tmp2.seismic,
                                                    outputs_tmp2.synth_seismic)
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = _similarity.dynamic_normalized_xcorr(outputs_tmp2.seismic,
                                                      outputs_tmp2.synth_seismic)
    else:
        xcorr = _similarity.prestack_traces_normalized_xcorr(outputs_tmp2.seismic,
                                                             outputs_tmp2.synth_seismic)
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = None

    outputs_tmp2.xcorr = xcorr
    outputs_tmp2.dxcorr = dxcorr

    return outputs_tmp2


def _intermediate_tie_v1(logset_md: grid.LogSet,
                         wellpath: grid.WellPath,
                         table: grid.TimeDepthTable,
                         seismic: grid.Seismic,
                         wavelet_extractor: wtie.learning.model.BaseEvaluator,
                         modeler: wtie.modeling.modeling.ModelingCallable,
                         parameters: dict
                         ) -> OutputSet:
    # Resampling
    seismic = _tie.resample_seismic(
        seismic, wavelet_extractor.expected_sampling)

    # Common steps
    logset_twt, seis_match, r_match = \
        _common_steps_tie_v1(logset_md,
                             wellpath,
                             table,
                             seismic,
                             wavelet_extractor,
                             modeler,
                             parameters)

    # (Zero-phased) unscaled wavelet
    wlt = _tie.compute_wavelet(seis_match, r_match,
                               modeler, wavelet_extractor,
                               zero_phasing=INTERMEDIATE_ZERO_PHASING,
                               scaling=False, expected_value=False)

    synth_seismic = _tie.compute_synthetic_seismic(modeler, wlt, r_match)

    return OutputSet(wlt, logset_twt, seis_match, synth_seismic,
                     wellpath, table, r_match)


def _search_best_params_v1(inputs: InputSet,
                           wavelet_extractor: wtie.learning.model.BaseEvaluator,
                           modeler: wtie.modeling.modeling.ModelingCallable,
                           search_space: dict,
                           num_iters: int,
                           random_ratio: float,
                           similarity_std: float
                           ) -> _optimizer.AxClient:
    # Resampling
    seismic = _tie.resample_seismic(
        inputs.seismic, wavelet_extractor.expected_sampling)

    # Optim client
    ax_client = _optimizer.create_ax_client(
        num_iters, random_ratio=random_ratio)

    ax_client.create_experiment(
        name="auto well tie",
        parameters=search_space,
        objective_name="goodness_of_match",
        minimize=False,
        choose_generation_strategy_kwargs=None
    )

    # Optimization
    print("Search for optimal parameters")
    sleep(1.0)
    for i in tqdm(range(num_iters)):
        try:
            h_params, trial_index = ax_client.get_next_trial()
        except RuntimeError:  # NotPSDError:
            print("Early stopping after %d/%d iterations." % (i+1, num_iters))
            break

        # table
        current_table = grid.TimeDepthTable.t_bulk_shift(inputs.table,
                                                         h_params['table_t_shift'])
        # common steps
        logset_twt, seis_match, r_match = \
            _common_steps_tie_v1(inputs.logset_md,
                                 inputs.wellpath,
                                 current_table,
                                 seismic,
                                 wavelet_extractor,
                                 modeler,
                                 h_params)

        # (zero-phased) unscaled wavelet
        current_wlt = _tie.compute_wavelet(seis_match, r_match, modeler,
                                           wavelet_extractor,
                                           zero_phasing=INTERMEDIATE_ZERO_PHASING,
                                           scaling=False,
                                           expected_value=INTERMEDIATE_EXPECTED_VALUE)

        # synthetic seismic
        synth_seismic = _tie.compute_synthetic_seismic(
            modeler, current_wlt, r_match)

        # similarity
        current_score = _similarity.central_xcorr_coeff(
            _tie.resample_seismic(seis_match, _tie.FINE_DT),
            _tie.resample_seismic(synth_seismic, _tie.FINE_DT)
        )

        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=(current_score, similarity_std))

    return ax_client


def _common_steps_tie_v1(logset_md: grid.LogSet,
                         wellpath: grid.WellPath,
                         table: grid.TimeDepthTable,
                         seismic: grid.seismic_t,
                         wavelet_extractor: wtie.learning.model.BaseEvaluator,
                         modeler: wtie.modeling.modeling.ModelingCallable,
                         params: dict):

    # log filtering
    logset_md = _tie.filter_md_logs(logset_md,
                                    median_size=params['logs_median_size'],
                                    threshold=params['logs_median_threshold'],
                                    std=params['logs_std'],
                                    std2=.8*params['logs_std'])

    # convertion
    logset_twt = _tie.convert_logs_from_md_to_twt(logset_md,
                                                  wellpath,
                                                  table,
                                                  wavelet_extractor.expected_sampling)

    # reflectivity
    r0 = _tie.compute_reflectivity(logset_twt, angle_range=seismic.angle_range)

    # matching
    seis_match, r0_match = _tie.match_seismic_and_reflectivity(seismic, r0)

    return logset_twt, seis_match, r0_match


#####################
# Default params
#####################
def get_default_search_space_v1():
    """
    Search space of version 1 is composed of 4 parameters:
        - "logs_median_size" : size (in number of samples) of the median filter window.
        - "logs_median_threshold" : threshold value with respect to the logs standard deviation.
        - "logs_std" : standard deviation of the gaussian smoothing filter.
        - "table_t_shift" : bulk shift in seconds of the depth-time relation table.

    Parameters are defined following the [Ax](https://github.com/facebook/Ax)
    documentation."""
    median_length_choice = dict(name="logs_median_size", type="choice",
                                values=[i for i in range(11, 73, 2)], value_type="int")

    median_th_choice = dict(name="logs_median_threshold", type="range",
                            bounds=[0.1, 5.5], value_type="float")

    std_choice = dict(name="logs_std", type="range",
                      bounds=[0.5, 6.5], value_type="float")

    table_t_shift_choice = dict(name="table_t_shift", type="range",
                                bounds=[-0.012, 0.012], value_type="float")

    search_space = [median_length_choice,
                    median_th_choice,
                    std_choice,
                    table_t_shift_choice,
                    ]
    return search_space
