"""Base dataset creation and handling."""


import h5py
import torch

from pathlib import Path
import numpy as np

import wtie
from wtie.utils.types_ import Dict, Tensor




############################
# BASE DATASET
############################
class BaseDataset:
    """In-memory data feeder."""
    def __init__(self,
                 data_creator: "SyntheticDataCreator"):

        # instance of SyntheticDataCreator
        self.data_creator = data_creator

        # load data in memory
        with h5py.File(data_creator.h5_file, 'r') as h5f:
            self.wavelets = h5f[data_creator.h5_group]['wavelets'][:]
            self.dt = h5f[data_creator.h5_group]['wavelets'].attrs['dt']
            self.wavelet_t = h5f[data_creator.h5_group]['wavelets'].attrs['t']

            self.reflectivities = h5f[data_creator.h5_group]['reflectivities'][:]
            self.seismics = h5f[data_creator.h5_group]['seismics'][:]

        assert self.wavelets.shape[0] == len(self)
        assert self.reflectivities.shape[0] == len(self)


        # gather some useful attributes
        self.wavelet_size = self.wavelets[0,:].size
        self.wavelet_duration = self.wavelet_size * self.dt
        self.reflectivity_size = self.reflectivities[0,:].size
        self.reflectivity_duration = self.reflectivity_size * self.dt # assumes same sampling as wavelets
        self.reflectivity_t = self.dt * np.arange(self.reflectivity_size)

    def __len__(self):
        return self.data_creator.num_examples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        wlt = self.wavelets[idx,:]
        ref = self.reflectivities[idx,:]

        # new noise everytime
        noise_ = self.data_creator.noise_callable()

        #seismic = self.data_creator.modeler(wavelet=wlt, reflectivity=ref, noise=_noise)
        seismic = self.seismics[idx,:]
        seismic += noise_

        return {'wavelet': wlt,
                'reflectivity': ref,
                'noise': noise_,
                'seismic': seismic
                }

    def prepare_for_network_input(self, data: np.ndarray) -> np.ndarray:
        data = data[np.newaxis,:]
        data = data[np.newaxis,:]
        return data.astype(np.float32)


################################
# PYTORCH DATASET
################################
class PytorchDataset(torch.utils.data.Dataset):
    """ """
    def __init__(self, base_dataset: BaseDataset):
        super().__init__()
        self.base = base_dataset


    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        data = self.base[idx]

        # convert numpy arrays to torch tensors and cast to <f4
        data_pt = {}
        for key, value in data.items():
            #add channel for input data
            if key in ['seismic', 'reflectivity']:
                value = value[np.newaxis,:]
            value = value.astype(np.float32)
            data_pt[key] = torch.from_numpy(value)

        return data_pt



############################
# SYNTHETIC DATASET CREATION
############################
class SyntheticDataCreator:
    def __init__(self,
                 num_examples: int,
                 rnd_wlt_callable: wtie.modeling.wavelet.RandomWaveletCallable,
                 rnd_ref_callable: wtie.modeling.reflectivity.RandomReflectivityCallable,
                 noise_callable: wtie.modeling.noise.NoiseCallable,
                 modeler: wtie.modeling.modeling.ModelingCallable,
                 h5_file: str,
                 h5_group: str,
                 from_scratch: bool=False):

        self.num_examples = num_examples

        self.wavelet_callable = rnd_wlt_callable
        self.reflectivity_callable = rnd_ref_callable
        self.noise_callable = noise_callable
        self.modeler = modeler

        self.h5_file = Path(h5_file)
        self.h5_group = h5_group

        # create dataset if does not exists or from_scratch
        if from_scratch and self.h5_file.is_file():
            self.h5_file.unlink()

        # create if file does not exists
        create = False
        if not self.h5_file.is_file():
            create = True

        # create if file exists but group does not
        else:
            with h5py.File(h5_file) as h5f:
                if h5_group not in h5f:
                    create = True


        if create:
            create_wavelet_dataset(num_examples, rnd_wlt_callable, h5_file, h5_group)
            create_reflectivity_dataset(num_examples, rnd_ref_callable, h5_file, h5_group)
            create_seismic_dataset(modeler, h5_file, h5_group)
        else:
            print("%s -> %s dataset already exists." % (h5_file, h5_group))


        assert self.h5_file.is_file()






###############################
# UTILS FUNCTIONS
###############################
def create_seismic_dataset(modeler: wtie.modeling.modeling.ModelingCallable,
                           h5_file: str,
                           h5_group: str
                           ) -> None:

    print("Create seismic traces from wavelets and reflectivities and save them in %s -> %s" % (h5_file, h5_group))

    with h5py.File(h5_file, 'r') as h5f:
        wavelets = h5f[h5_group]['wavelets'][:]
        #dt = h5f[h5_group]['wavelets'].attrs['dt']
        #wavelet_t = h5f[h5_group]['wavelets'].attrs['t']
        reflectivities = h5f[h5_group]['reflectivities'][:]

    seismics = np.zeros_like(reflectivities)
    for i in range(seismics.shape[0]):
        seismics[i,:] = modeler(wavelet=wavelets[i,:], reflectivity=reflectivities[i,:])


    # save to h5 file
    with h5py.File(h5_file, 'a') as h5f:
        group = h5f.require_group(h5_group)
        seis_ = group.create_dataset('seismics', data=seismics)



def create_wavelet_dataset(num_examples: int,
                           rnd_wlt_callable: wtie.modeling.wavelet.RandomWaveletCallable,
                           h5_file: str,
                           h5_group: str,
                           ) -> None:


    print("Create %d wavelets and save them in %s -> %s" % (num_examples,
                                                          h5_file,
                                                          h5_group))
    # initialization
    #simple_progressbar(0, num_examples-1)


    wlt_i = rnd_wlt_callable()
    t = wlt_i.t
    dt = wlt_i.dt
    n_samples = len(t)

    wlts = np.zeros((num_examples, n_samples))
    wlts[0] = wlt_i.y

    # creatre n wavelets
    #count = 1
    for i in range(1, num_examples):
        wlt_i = rnd_wlt_callable()
        wlts[i] = wlt_i.y
        #count += 1
        #simple_progressbar(count, num_examples-1)


    # save to h5 file
    with h5py.File(h5_file, 'a') as h5f:
        group = h5f.require_group(h5_group)
        wlts_ = group.create_dataset('wavelets', data=wlts)
        wlts_.attrs['dt'] = dt
        wlts_.attrs['t'] = t



def create_reflectivity_dataset(num_examples: int,
                                rnd_ref_callable: wtie.modeling.reflectivity.RandomReflectivityCallable,
                                h5_file: str,
                                h5_group: str
                                ) -> None:

    print("Create %d 1d reflectivities and save them in %s -> %s" % (num_examples,
                                                                    h5_file,
                                                                    h5_group))

    # initialization
    #simple_progressbar(0, num_examples-1)

    ref_i = rnd_ref_callable()
    n_samples = len(ref_i)

    refs = np.zeros((num_examples, n_samples))
    refs[0] = ref_i

    # creatre n wavelets
    #count = 1
    for i in range(1, num_examples):
        refs[i] = rnd_ref_callable()
        #count += 1
        #simple_progressbar(count, num_examples-1)

    # save to h5 file
    with h5py.File(h5_file, 'a') as h5f:
        group = h5f.require_group(h5_group)
        refs_ = group.create_dataset('reflectivities', data=refs)

