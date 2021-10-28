import numpy as np

from wtie.modeling import wavelet
from wtie.modeling import perturbations
from wtie.processing.taper import LinearTaper
from wtie.processing.sampling import Resampler
from wtie.modeling.reflectivity import (RandomSimplexReflectivityChooser,
                                        RandomUniformReflectivityChooser,
                                        RandomReflectivityCallable
                                        )
from wtie.modeling.modeling import convolution_modeling
from wtie.modeling.noise import RandomWhiteNoise


def test_synthetic_creation():
    # Wavelets
    dt = 0.004
    #dt_fine = 0.001

    #resampling_factor = int(dt // dt_fine)

    N = 32


    taper = LinearTaper(size=15)

    rand_rick_func = wavelet.RandomRickerTools(f_range=(12,44), dt=dt, n_samples=N)
    rand_butt_func = wavelet.RandomButterworthTools(lowcut_range=(3,15), highcut_range=(45,65),
                                                dt=dt, n_samples=N, order_range=(5,6))

    rand_om_func = wavelet.RandomOrmbyTools(f0_range=(1,7),
                                            f1_range=(5,15),
                                            f2_range=(25,40),
                                            f3_range=(30,80), dt=dt, n_samples=N)


    rand_base_wlt_funcs = [rand_butt_func, rand_rick_func, rand_om_func]

    ## perturbations
    perts = []
    perts.append(perturbations.RandomSimplexNoise(scale_range=(0.01,0.033),
                                               variation_scale_range=(N/45,
                                                                      N/25)))
    perts.append(perturbations.RandomConstantPhaseRotation(angle_range=(-30,30)))
    perts.append(perturbations.RandomTimeShift(max_samples=5))
    perts.append(perturbations.RandomIndependentPhaseRotation(angle_range=(-20,20)))
    perts.append(perturbations.RandomNotchFilter(dt=dt,freq_range=(8,55), band_range=(1,2), apply=True))
    perts.append(taper)
    perts.append(perturbations.RandomWhitexNoise(scale=0.005))

    perts = perturbations.Compose(perts, random_switch=True, p=.2)

    resampler = None #Resampler(current_dt=dt_fine, resampling_dt=dt)


    rdn_wlt_callable = wavelet.RandomWaveletCallable(\
                                random_base_wavelet_gens=rand_base_wlt_funcs,
                                perturbations=perts,
                                resampler=resampler)

    for _ in range(10):
        wlt = rdn_wlt_callable()

    # Reflectivity
    n_ref = 101
    #ref_t = np.arange(n_ref) * dt

    sparsity_rate_range = (0.4, 0.8)

    ## uniform
    rnd_u_chooser = RandomUniformReflectivityChooser(n_ref, sparsity_rate_range)

    ## simplex
    variation_scale_range = (10, 1000)

    rnd_s_chooser = RandomSimplexReflectivityChooser(n_ref,
                                                     sparsity_rate_range,
                                                     variation_scale_range)


    ## grouped
    rnd_ref_callable = RandomReflectivityCallable([rnd_u_chooser, rnd_s_chooser])
    for _ in range(10):
        ref = rnd_ref_callable()

    # Modeling
    noise_gen = RandomWhiteNoise(size=ref.size, scale_range=(0.1,0.2))
    for _ in range(10):
        noise = noise_gen()

    trace = convolution_modeling(wavelet=wlt.y, reflectivity=ref, noise=noise)

    assert trace.size == ref.size



