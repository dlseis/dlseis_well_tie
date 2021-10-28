from wtie.modeling import wavelet
from wtie.modeling import reflectivity
from wtie.modeling import perturbations

from wtie.processing.taper import CosSquareTaper
from wtie.modeling.noise import WhiteNoise
from wtie.modeling.modeling import ConvModeler



def get_wavelet_callable(dt=0.002, N=148):

    wavelet_size = N
    #resampling_factor = 2
    #dt_fine = dt / resampling_factor

    # resampler
    #resampler = Resampler(current_dt=dt_fine, resampling_dt=dt)

    # taper
    taper = CosSquareTaper(size=4)

    ## ricker
    fc_min = 12
    fc_max = 48
    rand_rick_func = wavelet.RandomRickerTools(f_range=(fc_min,fc_max),dt=dt,
                                       n_samples=wavelet_size)

    # butterworth
    order_range = (4,8)
    low_min = 2
    low_max = 5
    high_min = 35
    high_max = 70
    rand_butt_func = wavelet.RandomButterworthTools(lowcut_range=(low_min,low_max),
                                            highcut_range=(high_min,high_max),
                                            dt=dt,
                                            n_samples=wavelet_size,
                                            order_range=order_range)

    # composed
    rand_base_wlt_funcs = [rand_butt_func, rand_rick_func]

    # perturbations
    perts = []
    perts+=[perturbations.RandomSimplexNoise(scale=0.075, octave_range=(5,6), variation_scale=5.)] #[0.05,0.1]
    perts+=[perturbations.RandomConstantPhaseRotation(angle_range=(-35,35))]
    #perts+=[perturbations.RandomTimeShift(max_samples=4)]
    perts+=[perturbations.RandomIndependentPhaseRotation(angle_range=(-15,15))]
    perts+=[perturbations.RandomNotchFilter(dt=dt,freq_range=(50,65), band_range=(1,1))]
    #perts+=[perturbations.RandomWhitexNoise(scale=0.02)] #[0.01, 0.05]
    #perts+=[perturbations.RandomAmplitudeScaling(amplitude_range=(.6,1.4))]

    perts = perturbations.Compose(perts, random_switch=True, p=.2)


    # random callable
    rnd_wlt_callable = wavelet.RandomWaveletCallable(random_base_wavelet_gens=rand_base_wlt_funcs,
                                             perturbations=perts,
                                             taper=taper)


    return rnd_wlt_callable




def get_reflcetivity_callable(N=262):
    reflectivity_size = N
    sr_min = .4
    sr_max = .8


    # uniform
    rnd_u_chooser = reflectivity.RandomUniformReflectivityChooser(reflectivity_size,
                                                     sparsity_rate_range=(sr_min, sr_max))

    # simplex
    vs_min = 10
    vs_max = 2000
    rnd_s_chooser = reflectivity.RandomSimplexReflectivityChooser(reflectivity_size,
                                                     sparsity_rate_range=(sr_min, sr_max),
                                                     variation_scale_range=(vs_min, vs_max))

    # random callable
    rnd_ref_callable = reflectivity.RandomReflectivityCallable([rnd_u_chooser, rnd_s_chooser])

    return rnd_ref_callable














