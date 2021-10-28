import torch

import yaml

from pathlib import Path
import numpy as np

from wtie.modeling import wavelet
from wtie.modeling import reflectivity

from wtie.dataset import SyntheticDataCreator, BaseDataset, PytorchDataset
from wtie.modeling.noise import RandomWhiteNoise
from wtie.modeling.modeling import ConvModeler

from wtie.modeling.utils import create_or_load_synthetic_dataset




def test_tmp_path(tmp_path):
    assert Path(tmp_path).is_dir()




def test_dataset_creation_and_use(tmp_path):
    num_examples = 5

    # wavelets
    dt = 0.004
    N = 32
    rand_rick_func = wavelet.RandomRickerTools(f_range=(12,44), dt=dt, n_samples=N)
    rnd_wlt_callable = wavelet.RandomWaveletCallable(random_base_wavelet_gens=[rand_rick_func])

    # reflectivity
    n_ref = 101
    rnd_u_chooser = reflectivity.RandomUniformReflectivityChooser(n_ref, sparsity_rate_range=(0.4, 0.8))
    rnd_ref_callable = reflectivity.RandomReflectivityCallable([rnd_u_chooser])

    # noise
    noise_callable = RandomWhiteNoise(size=n_ref, scale_range=(0.05,0.15))

    # modeler
    modeler = ConvModeler()

    # files
    h5_file = Path(tmp_path) / 'test.h5'
    h5_group = 'test_group'

    # creation
    creator = SyntheticDataCreator(num_examples,
                                   rnd_wlt_callable,
                                   rnd_ref_callable,
                                   noise_callable,
                                   modeler,
                                   h5_file,
                                   h5_group,
                                   from_scratch=True)

    # base dataset
    base = BaseDataset(data_creator=creator)
    data = base[0]

    assert 'wavelet' in data
    assert 'reflectivity' in data
    assert 'noise' in data
    assert 'seismic' in data

    # pytorch dataset
    pt = PytorchDataset(base_dataset = base)
    data_pt = pt[0]

    assert type(data_pt['wavelet']) is torch.Tensor

    # test reload data
    creator2 = SyntheticDataCreator(num_examples,
                                   rnd_wlt_callable,
                                   rnd_ref_callable,
                                   noise_callable,
                                   modeler,
                                   h5_file,
                                   h5_group,
                                   from_scratch=False)

    # base dataset
    base2 = BaseDataset(data_creator=creator2)
    data2 = base2[0]

    assert np.allclose(data['wavelet'], data2['wavelet'])



def test_actual_data_creator(tmp_path):

    with open('../experiments/parameters.yaml', 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    params = parameters['synthetic_dataset']
    params['num_training_samples'] = 32
    params['num_validation_samples'] = 32

    save_dir = Path(tmp_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # files
    h5_file = save_dir / 'dataset.h5'
    h5_group_training = 'synth01/training'
    h5_group_validation = 'synth01/validation'

    training_creator, validation_creator = create_or_load_synthetic_dataset(h5_file,
                                                                            h5_group_training,
                                                                            h5_group_validation,
                                                                            params
                                                                            )

    assert True