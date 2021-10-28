import numpy as np
import torch

from pathlib import Path

from wtie.learning.network import Net
from wtie.learning.model import Evaluator, Model, VariationalModel
from wtie.modeling.modeling import ConvModeler

class DummyBaseDataset:
    """dummy data feeder."""
    def __init__(self):
        self.num_examples = 256
        self.dt = 0.004
        self.wavelet_size = 48
        self.wavelet_duration = 48 * self.dt
        self.reflectivity_size = 101
        self.reflectivity_duration = self.reflectivity_size * self.dt # assumes same sampling as wavelets


        self.wavelets = np.random.normal(size=(self.num_examples, self.wavelet_size)).astype(np.float32)
        self.reflectivities = np.random.normal(size=(self.num_examples, self.reflectivity_size)).astype(np.float32)

        self.modeler = ConvModeler()



    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx: int):
        wlt = self.wavelets[idx,:]
        ref = self.reflectivities[idx,:]

        _noise = np.zeros_like(ref)

        seismic = self.modeler(wavelet=wlt, reflectivity=ref)

        return {'wavelet': wlt,
                'reflectivity': ref,
                'noise': _noise,
                'seismic': seismic
                }

class DummyLogger:
    def write(self, message):
        pass


def test_network():
    # test network
    net = Net(wavelet_size=32)

    # [batch, channels, length]
    seismic = torch.from_numpy(np.random.normal(size=(4, 1, 101)).astype(np.float32))
    reflectivity = torch.from_numpy(np.random.normal(size=(4, 1, 101)).astype(np.float32))

    wavelet = net(seismic, reflectivity)

    wavelet = wavelet.detach().cpu().numpy()

    assert wavelet.shape[0] == 4
    assert wavelet.shape[1] == 32

    # test evaluator
    evaluator = Evaluator(network=net, expected_sampling=0.002)
    seismic = np.random.normal(size=(4, 1, 101)).astype(np.float32)
    reflectivity = np.random.normal(size=(4, 1, 101)).astype(np.float32)
    wavelet2 = evaluator(seismic, reflectivity)

    assert wavelet2.shape[0] == 4
    assert wavelet2.shape[1] == 32



def test_model(tmp_path):

    save_dir = Path(tmp_path)

    base_training = DummyBaseDataset()
    base_validation = DummyBaseDataset()

    params = dict(max_epochs=2,
                  learning_rate=0.0002,
                  lr_decay_rate=.5,
                  lr_decay_every_n_epoch=40,
                  batch_size=64,
                  weight_decay=0.01,
                  beta = 0.1,
                  alpha_init=0.3,
                  alpha_max=0.35,
                  alpha_scaling=1.1,
                  alpha_epoch_rate=10,
                  network_kwargs=None,
                  wavelet_loss_type='mae',
                  seismic_loss_type='mse',
                  min_delta_perc=5,
                  patience=10)


    for _Model in (Model, VariationalModel):

        model_instance = _Model(save_dir,
                           base_training,
                           base_validation,
                           params,
                           DummyLogger(),
                           tensorboard=None)

        model_instance.train()

    assert True





