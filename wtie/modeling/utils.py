"""Function to create synthetic data."""


from wtie.modeling.wavelet import (RandomRickerTools,
                                   RandomButterworthTools,
                                   RandomOrmbyTools,
                                   RandomWaveletCallable
                                   )
from wtie.processing.taper import CosinePowerTaper
from wtie.modeling import perturbations
from wtie.modeling.reflectivity import (RandomUniformReflectivityChooser,
                                        RandomSimplexReflectivityChooser,
                                        RandomBiUniformReflectivityChooser,
                                        RandomWeakUniformReflectivityChooser,
                                        RandomSpikeReflectivity,
                                        RandomReflectivityCallable
                                        )
from wtie.modeling.noise import RandomWhiteNoise
from wtie.modeling.modeling import ConvModeler

from wtie.dataset import SyntheticDataCreator

from wtie.utils.types_ import _path_t



def create_or_load_synthetic_dataset(h5_file : _path_t,
                                     h5_group_training: str,
                                     h5_group_validation: str,
                                     parameters: dict
                                     ):
    """TODO: find balance between parameters and hard-coding."""


    # alias
    params = parameters

    # Parameters
    dt = params['dt']
    #resampling_factor = 2
    #dt_fine = dt / resampling_factor


    num_training_samples = params['num_training_samples']
    num_validation_samples = params['num_validation_samples']


    #-------------
    # Wavelets
    #-------------
    wavelet_size = params['wavelet']['wavelet_size']

    # resampler
    #resampler = None #Resampler(current_dt=dt_fine, resampling_dt=dt)

    # taper
    taper = None #CosinePowerTaper(size=5, power=2)

    ## ricker
    fc_min = params['wavelet']['ricker']['fc_min']
    fc_max = params['wavelet']['ricker']['fc_max']
    rand_rick_func = RandomRickerTools(f_range=(fc_min,fc_max),dt=dt,
                                       n_samples=wavelet_size)

    # butterworth
    #order = params['wavelet']['butterworth']['order']
    order_min = params['wavelet']['butterworth']['order_min']
    order_max = params['wavelet']['butterworth']['order_max']
    low_min = params['wavelet']['butterworth']['low_min']
    low_max = params['wavelet']['butterworth']['low_max']
    high_min = params['wavelet']['butterworth']['high_min']
    high_max = params['wavelet']['butterworth']['high_max']
    rand_butt_func = RandomButterworthTools(lowcut_range=(low_min,low_max),
                                            highcut_range=(high_min,high_max),
                                            dt=dt,
                                            n_samples=wavelet_size,
                                            order_range=(order_min, order_max))

    # ormby
    f0_min = params['wavelet']['ormby']['f0_min']
    f0_max = params['wavelet']['ormby']['f0_max']
    f1_min = params['wavelet']['ormby']['f1_min']
    f1_max = params['wavelet']['ormby']['f1_max']
    f2_min = params['wavelet']['ormby']['f2_min']
    f2_max = params['wavelet']['ormby']['f2_max']
    f3_min = params['wavelet']['ormby']['f3_min']
    f3_max = params['wavelet']['ormby']['f3_max']
    rand_om_func = RandomOrmbyTools((f0_min, f0_max),
                                    (f1_min, f1_max),
                                    (f2_min, f2_max),
                                    (f3_min, f3_max),
                                    dt=dt, n_samples=wavelet_size)

    # composed
    rand_base_wlt_funcs = []
    if params['wavelet']['ricker']['use'] is True:
        rand_base_wlt_funcs.append(rand_rick_func)
    if params['wavelet']['butterworth']['use'] is True:
        rand_base_wlt_funcs.append(rand_butt_func)
    if params['wavelet']['ormby']['use'] is True:
        rand_base_wlt_funcs.append(rand_om_func)
    #rand_base_wlt_funcs = [rand_butt_func, rand_rick_func]

    # perturbations
    perts = []
    #perts += [CosinePowerTaper(size=7, power=3)]
    perts += [perturbations.RandomSimplexNoise(scale_range=(0.002,0.01),
                                               variation_scale_range=(wavelet_size/45, wavelet_size/25))]
    perts += [perturbations.RandomConstantPhaseRotation(angle_range=(-55,55))]
    #perts+=[perturbations.RandomTimeShift(max_samples=4)]
    perts += [perturbations.RandomSimplextPhaseRotation(scale_percentage_factor=3., max_abs_angle=30)]
    perts += [perturbations.RandomIndependentPhaseRotation(angle_range=(-1,1))] #(-30,30)
    #perts+=[perturbations.RandomNotchFilter(dt=dt,freq_range=(50,65), band_range=(1,1))]
    #perts+=[perturbations.RandomWhitexNoise(scale=0.02)] #[0.01, 0.05]
    #perts+=[perturbations.RandomAmplitudeScaling(amplitude_range=(.6,1.4))]
    #perts += [CosinePowerTaper(size=12, power=2)]

    perts = perturbations.Compose(perts, random_switch=False, p=.05)


    # random callable
    rnd_wlt_callable = RandomWaveletCallable(random_base_wavelet_gens=rand_base_wlt_funcs,
                                             perturbations=perts,
                                             taper=taper)
    #-------------
    # Reflectivity
    #-------------
    reflectivity_size = params['reflectivity']['reflectivity_size']
    sr_min = params['reflectivity']['sparsity_rate_min']
    sr_max = params['reflectivity']['sparsity_rate_max']
    max_min = params['reflectivity']['weak']['max_amplitude_min']
    max_max = params['reflectivity']['weak']['max_amplitude_max']


    # uniform
    rnd_u_chooser = RandomUniformReflectivityChooser(reflectivity_size,
                                                     sparsity_rate_range=(sr_min, sr_max))

    # weak uniform
    rnd_weak_u_chooser = RandomWeakUniformReflectivityChooser(reflectivity_size,
                                                              sparsity_rate_range=(sr_min, sr_max),
                                                              max_amplitude_range=(max_min,max_max))

    # simplex
    vs_min = params['reflectivity']['simplex']['variation_scale_min']
    vs_max = params['reflectivity']['simplex']['variation_scale_max']
    rnd_s_chooser = RandomSimplexReflectivityChooser(reflectivity_size,
                                                     sparsity_rate_range=(sr_min, sr_max),
                                                     variation_scale_range=(vs_min, vs_max))

    # bi-uniform
    rnd_biu_chooser = RandomBiUniformReflectivityChooser(reflectivity_size)

    # spike
    spike_ref = RandomSpikeReflectivity(reflectivity_size)

    # random callable
    rnd_ref_callable = RandomReflectivityCallable([rnd_u_chooser,
                                                   rnd_weak_u_chooser,
                                                   rnd_s_chooser,
                                                   rnd_biu_chooser,
                                                   spike_ref])


    #-------------
    # Noise
    #-------------
    nsc_min = params['noise']['scale_min']
    nsc_max = params['noise']['scale_max']
    noise_callable = RandomWhiteNoise(size=reflectivity_size, scale_range=(nsc_min,nsc_max))

    #-------------
    # Modeler
    #-------------
    params_modeler = None
    modeler = ConvModeler(kwargs=params_modeler)

    #-------------
    # Creators
    #-------------
    training_creator = SyntheticDataCreator(num_training_samples,
                                            rnd_wlt_callable,
                                            rnd_ref_callable,
                                            noise_callable,
                                            modeler,
                                            h5_file,
                                            h5_group_training,
                                            from_scratch=params['from_scratch'])

    validation_creator = SyntheticDataCreator(num_validation_samples,
                                              rnd_wlt_callable,
                                              rnd_ref_callable,
                                              noise_callable,
                                              modeler,
                                              h5_file,
                                              h5_group_validation,
                                              from_scratch=params['from_scratch'])



    return training_creator, validation_creator