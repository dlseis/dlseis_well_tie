"""Tools using global black-box optimization to improve wavelets"""

import numpy as np

from tqdm import tqdm


from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models as ax_Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

import wtie

from wtie.optimize import similarity as _similarity
from wtie.modeling.modeling import ModelingCallable
from wtie.utils.types_ import List, FunctionType
from wtie.processing import grid
from wtie.processing.spectral import zero_phasing as _zero_phasing
from wtie.processing.spectral import compute_spectrum as _compute_spectrum


def _preprocess_real_seismic(
    seismic: grid.Seismic, inverse_polarity: bool = False
) -> np.ndarray:
    # norm
    abs_max = max(seismic.values.max(), -seismic.values.min())
    new_seismic = seismic.values / abs_max
    if inverse_polarity:
        new_seismic *= -1.0
    return _prepare_for_input_to_network(new_seismic)


def _preprocess_reflectivity(reflectivity: grid.Reflectivity) -> np.ndarray:
    # norm
    abs_max = max(reflectivity.values.max(), -reflectivity.values.min())
    new_ref = reflectivity.values / abs_max
    return _prepare_for_input_to_network(new_ref)


def _prepare_for_input_to_network(x: np.ndarray) -> np.ndarray:
    # tensor shape
    x = x[np.newaxis, :]  # batch
    x = x[np.newaxis, :]  # channels
    return x.astype(np.float32)


def _get_wavelet_object(
    wavelet: np.ndarray, dt: float, name: str = None
) -> grid.Wavelet:

    duration = wavelet.size * dt
    # t = np.arange(-duration/2, (duration-dt)/2, dt)
    t = np.arange(-duration / 2, duration / 2, dt)
    return grid.Wavelet(wavelet, t, name=name)


def zero_phasing_wavelet(wavelet: grid.Wavelet) -> grid.Wavelet:
    wlt_0 = _zero_phasing(wavelet.values)
    return grid.Wavelet(wlt_0, wavelet.basis, name=wavelet.name)


def get_phase(wavelet: grid.Wavelet, to_degree: bool = True) -> np.ndarray:
    ff, ampl, power, phase = _compute_spectrum(
        wavelet.values, wavelet.sampling_rate, to_degree=to_degree
    )
    return ff, phase


def get_spectrum(wavelet: grid.Wavelet, to_degree: bool = True) -> np.ndarray:
    ff, ampl, power, phase = _compute_spectrum(
        wavelet.values, wavelet.sampling_rate, to_degree=to_degree
    )
    return ff, ampl, power, phase


def compute_expected_wavelet(
    evaluator: wtie.learning.model.BaseEvaluator,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    n_sampling: int = 50,
    inverse_polarity: bool = False,
    zero_phasing: bool = False,
) -> grid.Wavelet:

    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)
    assert np.allclose(seismic.sampling_rate, evaluator.expected_sampling, rtol=1e-3)

    seismic_ = _preprocess_real_seismic(seismic, inverse_polarity=inverse_polarity)
    ref_ = _preprocess_reflectivity(reflectivity)

    # Expected wavelet
    expected_wlt = evaluator.expected_wavelet(
        seismic=seismic_, reflectivity=ref_, squeeze=True
    )
    if inverse_polarity:
        expected_wlt *= -1.0

    expected_wlt = _get_wavelet_object(expected_wlt, seismic.sampling_rate)

    if zero_phasing:
        expected_wlt = zero_phasing_wavelet(expected_wlt)
    else:
        # Uncertainties
        wavelets = evaluator.sample_n_times(
            seismic=seismic_, reflectivity=ref_, n=n_sampling
        )
        amp_spectrums = []
        phase_spectrums = []
        for wlt in wavelets:
            if inverse_polarity:
                wlt *= -1.0
            wlt = _get_wavelet_object(wlt, seismic.sampling_rate)
            ff, ampl, power, phase = get_spectrum(wlt, to_degree=True)
            amp_spectrums.append(ampl)
            phase_spectrums.append(phase)
        amp_spectrums = np.stack(amp_spectrums, axis=0)
        phase_spectrums = np.stack(phase_spectrums, axis=0)
        uncertainties = grid.WaveletUncertainties(
            ff,
            np.mean(amp_spectrums, axis=0),
            np.std(amp_spectrums, axis=0),
            np.mean(phase_spectrums, axis=0),
            np.std(phase_spectrums, axis=0),
        )
        expected_wlt.uncertainties = uncertainties
    return expected_wlt


def compute_expected_prestack_wavelet(
    evaluator: wtie.learning.model.BaseEvaluator,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    zero_phasing: bool = False,
    inverse_polarity: bool = False,
) -> grid.PreStackWavelet:

    assert (seismic.angles == reflectivity.angles).all()

    wavelets = []
    for theta in seismic.angles:
        wlt_ = compute_expected_wavelet(
            evaluator,
            seismic[theta],
            reflectivity[theta],
            zero_phasing=zero_phasing,
            inverse_polarity=inverse_polarity,
        )
        wlt_.theta = theta
        wavelets.append(wlt_)
    return grid.PreStackWavelet(wavelets)


def grid_search_best_wavelet(
    evaluator: wtie.learning.model.BaseEvaluator,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    modeler: wtie.modeling.modeling.ModelingCallable,
    similarity: FunctionType,
    num_wavelets: int = 60,
    inverse_polarity: bool = False,
    zero_phasing: bool = False,
) -> grid.Wavelet:
    """Brut force search...
    Tow possible critera: best in terms of fit
    or best in terms of minimum absolute phase...
    """
    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)
    assert np.allclose(seismic.sampling_rate, evaluator.expected_sampling, rtol=1e-3)

    seismic_ = _preprocess_real_seismic(seismic, inverse_polarity=inverse_polarity)
    ref_ = _preprocess_reflectivity(reflectivity)

    wavelets = evaluator.sample_n_times(
        seismic=seismic_, reflectivity=ref_, n=num_wavelets
    )
    # uncertainties
    amp_spectrums = []
    phase_spectrums = []
    for wlt in wavelets:
        wlt = _get_wavelet_object(wlt, seismic.sampling_rate)
        ff, ampl, power, phase = get_spectrum(wlt, to_degree=True)
        amp_spectrums.append(ampl)
        phase_spectrums.append(phase)
    amp_spectrums = np.stack(amp_spectrums, axis=0)
    phase_spectrums = np.stack(phase_spectrums, axis=0)
    uncertainties = grid.WaveletUncertainties(
        ff,
        np.mean(amp_spectrums, axis=0),
        np.std(amp_spectrums, axis=0),
        np.mean(phase_spectrums, axis=0),
        np.std(phase_spectrums, axis=0),
    )

    pack = []

    for i in range(len(wavelets)):
        wlt = _get_wavelet_object(wavelets[i], seismic.sampling_rate)
        if zero_phasing:
            wlt = zero_phasing_wavelet(wlt)
        synth_seismic = compute_synthetic_seismic(modeler, wlt, reflectivity)
        current_score = similarity(seismic.values, synth_seismic.values)

        # only look at phase before 60° for stability reasons
        ff, current_phase = get_phase(wlt, to_degree=True)
        valid_idx = np.argmin(np.abs(ff - 60))
        current_mean_abs_phase = np.mean(np.abs(current_phase[:valid_idx]))
        pack.append((wlt, current_score, current_mean_abs_phase))

    # sort list based on similarity score
    pack.sort(key=lambda a: a[1], reverse=True)
    # print("FIRST")
    # for p in pack:
    #    print(p[1],p[2])

    # take 10% best
    ntop = int(0.1 * num_wavelets)
    pack = pack[:ntop]

    # take smallest absolute phase amoung top scores
    if not zero_phasing:
        pack.sort(key=lambda a: a[2], reverse=False)
        # print("SECOND")
        # for p in pack:
        #    print(p[1],p[2])

    best_wavelet = pack[0][0]
    # print("FINAL")
    # print(pack[0][1],pack[0][2])

    if inverse_polarity:
        best_wavelet *= -1.0

    # set uncertainites
    best_wavelet.uncertainties = uncertainties

    return best_wavelet


def grid_search_best_prestack_wavelet(
    evaluator: wtie.learning.model.BaseEvaluator,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    modeler: wtie.modeling.modeling.ModelingCallable,
    similarity: FunctionType,
    num_wavelets: int = 60,
    zero_phasing: bool = False,
) -> grid.Wavelet:
    """ """
    assert (seismic.angles == reflectivity.angles).all()

    best_wavelet = []
    for i in range(seismic.num_traces):
        wlt = grid_search_best_wavelet(
            evaluator,
            seismic.traces[i],
            reflectivity.traces[i],
            modeler,
            similarity,
            num_wavelets=num_wavelets,
            zero_phasing=zero_phasing,
        )
        wlt.theta = seismic.angles[i]
        best_wavelet.append(wlt)

    return grid.PreStackWavelet(best_wavelet)


def compute_synthetic_seismic(
    modeler: wtie.modeling.modeling.ModelingCallable,
    wavelet: grid.Wavelet,
    reflectivity: grid.Reflectivity,
) -> grid.Seismic:

    assert np.allclose(wavelet.sampling_rate, reflectivity.sampling_rate)
    assert reflectivity.is_twt

    seismic = modeler(wavelet.values, reflectivity.values)
    return grid.Seismic(seismic, reflectivity.basis, "twt", name="Synthetic seismic")


def compute_synthetic_prestack_seismic(
    modeler: wtie.modeling.modeling.ModelingCallable,
    wavelet: grid.PreStackWavelet,
    reflectivity: grid.PreStackReflectivity,
) -> grid.PreStackSeismic:

    assert (wavelet.angles == reflectivity.angles).all()

    seismics = []
    for theta in wavelet.angles:
        seis_ = compute_synthetic_seismic(modeler, wavelet[theta], reflectivity[theta])
        seis_.theta = theta
        seismics.append(seis_)
    return grid.PreStackSeismic(seismics)


############################################################
############################################################


def select_best_wavelet(
    wavelets: List[np.ndarray],
    seismic: np.ndarray,
    reflectivity: np.ndarray,
    modeler: ModelingCallable,
    num_iters: int = None,
    noise_perc: float = 10,
):
    """ """
    raise NotImplementedError()

    if num_iters is None:
        # overkill but its cheap
        num_iters = int(len(wavelets) * 1.2)

    n_sobol = int(0.7 * num_iters)  # 70%
    n_bayes = num_iters - n_sobol  # 30%

    ax_gen_startegy = GenerationStrategy(
        [
            GenerationStep(ax_Models.SOBOL, num_trials=n_sobol),
            GenerationStep(ax_Models.BOTORCH, num_trials=n_bayes),
        ]
    )

    ax_client = AxClient(generation_strategy=ax_gen_startegy, verbose_logging=False)

    choice = dict(
        name="wavelet_choice",
        type="range",
        bounds=[0, len(wavelets) - 1],
        value_type="int",
    )

    search_space = [choice]

    # Maximization
    ax_client.create_experiment(
        name="wavelet_distribution_choice",
        parameters=search_space,
        objective_name="choice_performance",
        minimize=False,
        choose_generation_strategy_kwargs=None,
    )

    noise_level = (noise_perc / 100) * np.std(seismic)
    noise1_ = np.random.normal(scale=noise_level, size=seismic.shape)
    noise2_ = np.random.normal(scale=noise_level, size=seismic.shape)
    xcorr_coeff_error = 1.0 - _similarity.normalized_xcorr_maximum(
        seismic + noise1_, seismic + noise2_
    )

    for i in tqdm(range(num_iters)):
        h_params, trial_index = ax_client.get_next_trial()
        current_choice = h_params["wavelet_choice"]
        current_wavelet = wavelets[current_choice]

        current_seismic = modeler(
            wavelet=current_wavelet, reflectivity=reflectivity, noise=None
        )

        current_coeff = _similarity.normalized_xcorr_maximum(current_seismic, seismic)

        ax_client.complete_trial(
            trial_index=trial_index, raw_data=(current_coeff, xcorr_coeff_error)
        )

    best_choice = ax_client.get_best_parameters()[0]["wavelet_choice"]

    return wavelets[best_choice], ax_client


def scale_wavelet(
    wavelet: grid.Wavelet,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    modeler: ModelingCallable,
    min_scale: float = 0.01,
    max_scale: float = 100,
    num_iters: int = 80,
    noise_perc: float = 5,
    is_tqdm: bool = True,
):
    """
    TODO: refactor like scale_prestack_wavelet"""

    n_sobol = int(0.65 * num_iters)  # 65%
    n_bayes = num_iters - n_sobol  # 35%

    ax_gen_startegy = GenerationStrategy(
        [
            GenerationStep(ax_Models.SOBOL, num_trials=n_sobol),
            GenerationStep(ax_Models.BOTORCH, num_trials=n_bayes),
        ]
    )

    ax_client = AxClient(generation_strategy=ax_gen_startegy, verbose_logging=False)

    scaler = dict(
        name="scaler", type="range", bounds=[min_scale, max_scale], value_type="float"
    )

    search_space = [scaler]

    ax_client.create_experiment(
        name="wavelet_absolute_scale_estimation",
        parameters=search_space,
        objective_name="scaling_loss",
        minimize=True,
        choose_generation_strategy_kwargs=None,
    )

    noise_level = (noise_perc / 100) * np.std(seismic.values)
    seismic_energy = _similarity.energy(seismic.values)

    for i in tqdm(range(num_iters), disable=(not is_tqdm)):
        h_params, trial_index = ax_client.get_next_trial()
        current_scaler = h_params["scaler"]
        current_wavelet_ = current_scaler * wavelet.values

        current_seismic = modeler(
            wavelet=current_wavelet_, reflectivity=reflectivity.values, noise=None
        )

        current_energy = _similarity.energy(current_seismic)

        # error = np.abs(current_energy - seismic_energy)
        error = np.abs(current_energy / seismic_energy - 1.0)

        # TODO: IS THIS CORRECT?
        error_std = _similarity.energy(
            np.random.normal(scale=noise_level, size=seismic.shape)
        )

        ax_client.complete_trial(trial_index=trial_index, raw_data=(error, error_std))

    best_scaler = ax_client.get_best_parameters()[0]["scaler"]

    scaled_wlt = grid.Wavelet(
        best_scaler * wavelet.values,
        wavelet.basis,
        name=wavelet.name,
        uncertainties=wavelet.uncertainties,
        theta=wavelet.theta,
    )

    return scaled_wlt, ax_client


def scale_prestack_wavelet(
    wavelet: grid.PreStackWavelet,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    modeler: ModelingCallable,
    min_scale: float = 0.01,
    max_scale: float = 100,
    num_iters: int = 50,
    noise_perc: float = 5,
):

    assert (wavelet.angles == seismic.angles).all()
    assert (wavelet.angles == reflectivity.angles).all()

    scaled_wavelet = []
    ax_clients = []

    for i in tqdm(range(wavelet.num_traces)):
        wlt, ax_client = scale_wavelet(
            wavelet.traces[i],
            seismic.traces[i],
            reflectivity.traces[i],
            modeler,
            min_scale=min_scale,
            max_scale=max_scale,
            num_iters=num_iters,
            noise_perc=noise_perc,
            is_tqdm=False,
        )

        wlt.theta = wavelet.angles[i]

        scaled_wavelet.append(wlt)
        ax_clients.append(ax_client)

    return grid.PreStackWavelet(scaled_wavelet, name=wavelet.name), ax_clients
