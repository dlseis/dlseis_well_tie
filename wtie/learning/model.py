import time
import pickle

from pathlib import Path
from multiprocessing import cpu_count

from tqdm import tqdm

import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np


from wtie.learning.losses import ReconstructionLoss, VariationalLoss
from wtie.dataset import PytorchDataset
from wtie.learning.network import Net, VariationalNetwork
from wtie.learning.utils import EarlyStopping, AlphaScheduler
from wtie.utils.types_ import tensor_or_ndarray, _path_t, List
from wtie.dataset import BaseDataset
from wtie.utils.logger import Logger



###############################################
# BASE
###############################################
class BaseModel:
    """ """

    # some names
    default_trained_net_state_dict_name = 'trained_net_state_dict.pt'


    def __init__(self, save_dir: _path_t,
                       base_train_dataset: BaseDataset,
                       base_val_dataset: BaseDataset,
                       parameters: dict,
                       logger: Logger,
                       device: torch.device=None,
                       tensorboard: SummaryWriter=None,
                       save_checkpoints: bool=False):

        self.start_time = time.time()

        # work directory
        self.save_dir = Path(save_dir)
        assert self.save_dir.is_dir()
        self._save_ckpt = save_checkpoints

        # logger and tb
        self.logger = logger
        self.tensorboard = tensorboard


        # parameters
        self.params = parameters
        self.start_epoch = 0
        self.current_epoch = 0
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.max_epochs = parameters['max_epochs']


        # datasets No need to add to member attributes
        #self.base_train_dataset = base_train_dataset
        #self.base_val_dataset = base_val_dataset

        logger.write(("Start training in directory %s") % self.save_dir)

        # dataloaders
        pt_train_dataset = PytorchDataset(base_train_dataset)
        pt_val_dataset = PytorchDataset(base_val_dataset)

        self.num_training_samples = len(base_train_dataset)
        self.num_validation_samples = len(base_val_dataset)

        self.train_loader = torch.utils.data.DataLoader(pt_train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=min(6,cpu_count()),
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(pt_val_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=min(4,cpu_count()),
                                                      pin_memory=True)


        # net and stuff
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.write("Computing device: %s" % str(self.device))


        # to be orverwriten in child class
        self.early_stopping = None
        self.schedulers: list = []




    def train_one_epoch(self):
        raise NotImplementedError()

    def validate_training(self):
        raise NotImplementedError()

    def train(self):
        _div = self.num_training_samples // self.batch_size
        _remain = int(self.num_training_samples % self.batch_size > 0)
        num_iterations_per_epoch = _div + _remain
        self.logger.write(\
            ("Training network for %d epochs (%d iterations per epoch)" % \
             (self.max_epochs, num_iterations_per_epoch))
            )

        is_early_stop = False
        for epoch in tqdm(range(self.start_epoch, self.max_epochs)):
            # training / validation
            self.train_one_epoch()
            current_val_loss = self.validate_training()

            # schedulers
            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()
            #self.scheduler.step()

            # increment count
            self.current_epoch += 1

            # monitor early stopping critera
            if self.early_stopping is not None:
                is_early_stop = self.early_stopping.step(current_val_loss)
                if is_early_stop:
                    break

            # tb
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("lr",
                                            self.scheduler.get_last_lr()[0],
                                            self.current_epoch)

            # ckpt
            if self._save_ckpt:
                if (epoch % (self.max_epochs // 4) == 0) or (epoch == self.max_epochs - 1):
                    ckpt_path = self.save_dir / ("ckpt_epoch%s.tar" % str(epoch+1).zfill(3))
                    self.save_model_ckpt(ckpt_path, epoch)


        if is_early_stop:
            self.logger.write(("Early stopping at epoch %d" % epoch))

        # save
        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()
        self.save_network(self.save_dir / Model.default_trained_net_state_dict_name)


        # tb
        if self.tensorboard:
            self.tensorboard.flush()



    def save_history(self):
        with open(self.save_dir / 'history.pkl', "wb") as fp:
            pickle.dump(self.history, fp)


    def save_network(self,path):
        """Prefered extension is .pt """
        self.logger.write("Saving network's state_dict to %s" % path)
        torch.save(self.net.state_dict(), path)

    def restore_network_from_state_dict(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


    def save_model_ckpt(self, path, epoch):
        """Prefered extension is .tar """
        self.logger.write("Saving model's checkpoint to %s" % path)
        torch.save({
            'epoch': epoch,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
            }, path)

    def restore_model_from_ckpt(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.current_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']



#########################################
# MODELS
#########################################


class Model(BaseModel):
    """ """

    # some names
    default_trained_net_state_dict_name = BaseModel.default_trained_net_state_dict_name


    def __init__(self, save_dir,
                       base_train_dataset,
                       base_val_dataset,
                       parameters,
                       logger,
                       device=None,
                       tensorboard=None,
                       save_checkpoints=False):

        super().__init__(save_dir,
                         base_train_dataset,
                         base_val_dataset,
                         parameters,
                         logger,
                         device=device,
                         tensorboard=tensorboard,
                         save_checkpoints=save_checkpoints)


        if parameters['network_kwargs'] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters['network_kwargs']



        self.net = Net(base_train_dataset.wavelet_size,
                       network_kwargs)
        self.net.to(self.device)


        if self.tensorboard is not None:
            self.tensorboard.add_graph(self.net, next(iter(self.train_loader)).to(self.device))

        self.loss = ReconstructionLoss(parameters)


        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-08, amsgrad=False,
                                          weight_decay=parameters['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         parameters['lr_decay_every_n_epoch'],
                                                         gamma=parameters['lr_decay_rate'])

        self.schedulers = [lr_scheduler]

        self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
                                        patience=parameters['patience'],
                                        min_epochs=int(0.8*self.max_epochs))

        self.history = {}
        for key in self.loss.key_names:
            self.history['train_loss_' + key] = []
            self.history['val_loss_' + key] = []



    def train_one_epoch(self):
        self.net.train()

        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0


        count_loop = 0
        for data_batch in self.train_loader:
            count_loop += 1
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}# to gpu

            wavelet_output_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])

            loss = self.loss(data_batch, wavelet_output_batch)
            loss['total'].backward() # backprop
            self.optimizer.step() # update params

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()


        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history['train_loss_' + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/train/" + key,
                                            _avg_numeric_loss,
                                            self.current_epoch)




    def validate_training(self):
        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0


        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_output_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])
                loss = self.loss(data_batch, wavelet_output_batch)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()


        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history['val_loss_' + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/val/" + key,
                                            _avg_numeric_loss, self.current_epoch)



        return loss_numerics['total'] / count_loop








class VariationalModel(BaseModel):
    """ """

    # some names
    default_trained_net_state_dict_name = BaseModel.default_trained_net_state_dict_name


    def __init__(self, save_dir,
                       base_train_dataset,
                       base_val_dataset,
                       parameters,
                       logger,
                       device=None,
                       tensorboard=None,
                       save_checkpoints=False):

        super().__init__(save_dir,
                         base_train_dataset,
                         base_val_dataset,
                         parameters,
                         logger,
                         device=device,
                         tensorboard=tensorboard,
                         save_checkpoints=save_checkpoints)


        if parameters['network_kwargs'] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters['network_kwargs']



        self.net = VariationalNetwork(base_train_dataset.wavelet_size,
                       network_kwargs)
        self.net.to(self.device)


        if self.tensorboard is not None:
            self.tensorboard.add_graph(self.net, next(iter(self.train_loader)).to(self.device))

        self.loss = VariationalLoss(parameters)


        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-08, amsgrad=False,
                                          weight_decay=parameters['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         parameters['lr_decay_every_n_epoch'],
                                                         gamma=parameters['lr_decay_rate'])

        self.alpha_scheduler = AlphaScheduler(loss=self.loss,
                                              alpha_init=parameters['alpha_init'],
                                              alpha_max=parameters['alpha_max'],
                                              rate=parameters['alpha_scaling'],
                                              every_n_epoch=parameters['alpha_epoch_rate'])

        self.schedulers = [lr_scheduler, self.alpha_scheduler]

        #self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
        #                                patience=parameters['patience'],
        #                                min_epochs=int(0.8*self.max_epochs))

        self.history = {}
        for key in self.loss.key_names:
            self.history['train_loss_' + key] = []
            self.history['val_loss_' + key] = []


    def train_one_epoch(self):
        self.net.train()

        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0


        count_loop = 0
        for data_batch in self.train_loader:
            count_loop += 1
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}# to gpu

            wavelet_batch, mu_batch, log_var_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])

            loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)
            loss['total'].backward() # backprop
            self.optimizer.step() # update params

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()


        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history['train_loss_' + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/train/" + key,
                                            _avg_numeric_loss, self.current_epoch)



    def validate_training(self):
        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0


        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_batch, mu_batch, log_var_batch = self.net(\
                                            seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity']
                                            )
                loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()


        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history['val_loss_' + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/val/" + key,
                                            _avg_numeric_loss,
                                            self.current_epoch)

        return loss_numerics['total'] / count_loop



#####################################
# LIGHT WEIGHT FOR HYPER-OTPIM
#####################################
class BaseLightModel:
    """For hyper-parameters search."""

    def __init__(self, base_train_dataset,
                       base_val_dataset,
                       parameters,
                       device=None):

        # parameters
        self.params = parameters
        self.current_epoch = 0
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.max_epochs = parameters['max_epochs']


        # dataloaders
        pt_train_dataset = PytorchDataset(base_train_dataset)
        pt_val_dataset = PytorchDataset(base_val_dataset)

        self.num_training_samples = len(base_train_dataset)
        self.num_validation_samples = len(base_val_dataset)

        self.train_loader = torch.utils.data.DataLoader(pt_train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=min(4,cpu_count()),
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(pt_val_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=min(2,cpu_count()),
                                                      pin_memory=True)


        # net and stuff
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # to overwtite in children class
        self.early_stopping = None
        self.schedulers: list = []


    def train_and_validate(self):
        count = 0
        #simple_progressbar(count, self.max_epochs)
        for epoch in range(0, self.max_epochs):
            self.train_one_epoch()
            current_val_loss, _ = self.validate_training()

            #self.lr_scheduler.step()
            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()

            self.current_epoch += 1
            count += 1

            #simple_progressbar(count, self.max_epochs)
            if self.early_stopping is not None:
                if self.early_stopping.step(current_val_loss):
                    break

        #print("Trained for %d/%d epochs" % (epoch+1, self.max_epochs))
        return self.validate_training()



class LightModel(BaseLightModel):

    def __init__(self, base_train_dataset,
                       base_val_dataset,
                       parameters,
                       device=None):

        super().__init__(base_train_dataset,
                         base_val_dataset,
                         parameters,
                         device=device)



        if parameters['network_kwargs'] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters['network_kwargs']



        self.net = Net(base_train_dataset.wavelet_size,
                       network_kwargs)
        self.net.to(self.device)


        self.loss = ReconstructionLoss(parameters)


        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-08, amsgrad=False,
                                          weight_decay=parameters['weight_decay'])

        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #optimizer=self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, cooldown=0, min_lr=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         parameters['lr_decay_every_n_epoch'],
                                                         gamma=parameters['lr_decay_rate'])


        self.schedulers = [lr_scheduler]

        self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
                                        patience=parameters['patience'],
                                        min_epochs=int(0.8*self.max_epochs))




    def train_one_epoch(self):
        self.net.train()

        for data_batch in self.train_loader:
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}# to gpu

            wavelet_output_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])

            loss = self.loss(data_batch, wavelet_output_batch)
            loss['total'].backward() # backprop
            self.optimizer.step() # update params



    def validate_training(self):
        total_validation_error_mean = 0.
        total_validation_error_std = 0.
        count_loop = 0

        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_output_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])
                loss = self.loss(data_batch, wavelet_output_batch)
                total_validation_error_mean += loss['validation_error_mean'].item()
                total_validation_error_std += loss['validation_error_std'].item() # ~approximation

        mean = total_validation_error_mean / count_loop
        std = total_validation_error_std / count_loop
        return mean, std





class LightVariationalModel(BaseLightModel):

    def __init__(self, base_train_dataset,
                       base_val_dataset,
                       parameters,
                       device=None):

        super().__init__(base_train_dataset,
                         base_val_dataset,
                         parameters,
                         device=device)



        if parameters['network_kwargs'] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters['network_kwargs']



        self.net = VariationalNetwork(base_train_dataset.wavelet_size,
                       network_kwargs)
        self.net.to(self.device)


        self.loss = VariationalLoss(parameters)


        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-08, amsgrad=False,
                                          weight_decay=parameters['weight_decay'])

        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #optimizer=self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, cooldown=0, min_lr=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         parameters['lr_decay_every_n_epoch'],
                                                         gamma=parameters['lr_decay_rate'])
        self.schedulers = [lr_scheduler]
        #self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
                                        #patience=parameters['patience'],
                                        #min_epochs=int(0.8*self.max_epochs))




    def train_one_epoch(self):
        self.net.train()

        for data_batch in self.train_loader:
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}# to gpu

            wavelet_batch, mu_batch, log_var_batch = self.net(seismic=data_batch['seismic'],
                                            reflectivity=data_batch['reflectivity'])

            loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)
            loss['total'].backward() # backprop
            self.optimizer.step() # update params



    def validate_training(self):
        #TODO: ugly...

        recon_error_mean = 0.
        recon_error_std = 0.
        var_error_mean = 0.
        var_error_std = 0.

        count = 0

        return_validation_dict = {}

        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_batch, mu_batch, log_var_batch = \
                    self.net(seismic=data_batch['seismic'],
                             reflectivity=data_batch['reflectivity'])
                loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)

                val_error_dict = loss['validation_error']

                recon_error_mean += val_error_dict['reconstruction_error'][0].item()
                recon_error_std += val_error_dict['reconstruction_error'][1].item() # ~approximation

                var_error_mean += val_error_dict['variation_error'][0].item()
                var_error_std += val_error_dict['variation_error'][1].item() # ~approximation


        return_validation_dict['reconstruction_error'] = \
            (recon_error_mean / count, recon_error_std / count)

        return_validation_dict['variation_error'] = \
            (var_error_mean / count, var_error_std / count)

        ##############
        #TMP
        #print("Reconstruction error (mean, std): ", return_validation_dict['reconstruction_error'])
        #print("Variation error (mean, std): ", return_validation_dict['variation_error'])
        #print("\n")

        # api: https://ax.dev/docs/trial-evaluation.html
        # Dict[Tuple[mean: float, std: float]]
        return return_validation_dict








###############################
# EVALUATORS
###############################

class BaseEvaluator:
    """Lighweigth class to peform evaluation (compute a wavelet given seismic and reflectivity)"""
    def __init__(self, network: torch.nn.Module,
                 expected_sampling: float,
                 state_dict: str=None,
                 device: torch.device=None,
                 verbose: bool=True):
        """Patameters
        -------------
        network : pytorch nn.Module
        expected_sampling: sampling rate in seconds of the input reflectivity and seismic.
        """

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.expected_sampling = expected_sampling


        self.net = network
        self.net.to(self.device)


        self.state_dict = state_dict
        if state_dict is not None:
            if verbose:
                print("Loading network parameters from %s" % state_dict)
            self.net.load_state_dict(torch.load(state_dict, map_location=self.device))
        else:
            if verbose:
                print("Network initialized randomly.")



class VariationalEvaluator(BaseEvaluator):
    """Compute a wavelet given seismic and reflectivity"""
    def __init__(self, network, expected_sampling,state_dict=None,
                 device=None, verbose=True):

        super().__init__(network=network,
                         expected_sampling=expected_sampling,
                         state_dict=state_dict,
                         device=device,
                         verbose=verbose)


    def expected_wavelet(self,
                         seismic: tensor_or_ndarray,
                         reflectivity: tensor_or_ndarray,
                         squeeze: bool = True,
                         ) -> np.ndarray:

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net.compute_expected_wavelet(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        return wavelet


    def sample(self,
               seismic: tensor_or_ndarray,
               reflectivity: tensor_or_ndarray,
               squeeze: bool = True,
               ) -> np.ndarray:

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net.sample(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        return wavelet


    def sample_n_times(self,
               seismic: tensor_or_ndarray,
               reflectivity: tensor_or_ndarray,
               n: int,
               squeeze: bool = True,
               ) -> List[np.ndarray]:

        wavelets = []

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)

            for _ in range(n):
                wavelet_i = self.net.sample(seismic, reflectivity)
                wavelet_i = wavelet_i.cpu().data.numpy()

                if squeeze:
                    wavelet_i = np.squeeze(wavelet_i)

                    wavelets.append(wavelet_i)

        return wavelets






class Evaluator(BaseEvaluator):
    """Compute a wavelet given seismic and reflectivity"""
    def __init__(self, network, expected_sampling, state_dict=None,
                 device=None, verbose=True):

        super().__init__(network=network,
                         expected_sampling=expected_sampling,
                         state_dict=state_dict,
                         device=device,
                         verbose=verbose)

    def __call__(self,
                 seismic: tensor_or_ndarray,
                 reflectivity: tensor_or_ndarray,
                 squeeze: bool = False,
                 scale_factor: float=None
                 ) -> np.ndarray:

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        if scale_factor is not None:
            wavelet *= scale_factor

        return wavelet












