"""TODO: make the code more flexible to a change of architecture."""

# See https://github.com/AntixK/PyTorch-VAE

import torch
import torch.nn as nn
import torch.nn.functional as F

from wtie.learning.blocks import DoubleConv1d, Down1d, LinearLrelu

from wtie.utils.types_ import List, Tuple, Tensor




##################################
# Constants
##################################
#_KERNEL_SIZE = 3
#_PADDING = _KERNEL_SIZE // 2
#_DOWN_SAMPLE_FACTOR = 2

#_ARGS = (_DOWN_SAMPLE_FACTOR, _KERNEL_SIZE, _PADDING)



##################################
# Abstarct Network
##################################

# TODO all nets must have same init arguments

##################################
# Regular Network
##################################

class Network(nn.Module):
    def __init__(self,
                 wavelet_size: int,
                 params: dict=None) -> None:

        super().__init__()

        self.wavelet_size = wavelet_size

        # Get extra paramters
        if params is None:
            params = {}

        p_drop = params.get('dropout_rate', .25)
        k_size = params.get('kernel_size', 5)
        padding = k_size // 2#params.get('kernel_size', k_size//2)
        downsampling_factor = params.get('downsampling_factor', 2)

        n_in_channels = 2


        # ENCODE
        se_params = dict(factor=downsampling_factor, kernel_size=k_size, padding=padding)
        modules = []
        modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
        modules += [nn.Dropout(p=p_drop, inplace = True)]
        modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
        modules += [nn.Dropout(p=p_drop, inplace = True)]
        modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
        modules += [nn.Dropout(p=p_drop, inplace = True)]
        modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        self.encoder = Sequential(modules)



        # DECODE
        modules = []
        modules += [LinearLrelu(in_features=256,out_features=512)]
        modules += [nn.Dropout(p=p_drop, inplace = True)]
        #modules += [LinearLrelu(in_features=n_hidden,out_features=n_hidden)]
        #modules += [nn.Dropout(p=p_drop, inplace = True)]
        modules += [nn.Linear(512, wavelet_size)]
        self.wavelet_decoder = Sequential(modules)




        # LEARN AMPLITUDE
        #modules = []
        #modules += [LinearLrelu(in_features=latent_dim,out_features=1)]
        #self.scaler = Sequential(modules)
        self.scaling = ScalingOperator()





    def encode(self, seismic: Tensor, reflecticvity: Tensor) -> Tensor:
        """
        Concatenates inputs in  channel dims and
        encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x L]
        :return: (Tensor) List of latent codes
        """
        # cat
        x = torch.cat((seismic, reflecticvity), dim=1)

        # encode
        x = self.encoder(x)

        # gap
        x = torch.mean(x, 2, keepdim=False)

        return x



    def decode_wavelet(self, z: Tensor) -> Tensor:
        w = self.wavelet_decoder(z)
        w = self.scaling(w)
        return w
        #scale = self.scaler(z)
        #return w*scale




    def forward(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        z = self.encode(seismic, reflectivity)
        wavelet = self.decode_wavelet(z)
        return wavelet



# alias
Net = Network




###################################
# Variational Network
###################################

class VariationalNetwork(nn.Module):
    def __init__(self,
                 wavelet_size: int,
                 params: dict=None,
                 one_ms: bool=False) -> None:

        super().__init__()

        self.wavelet_size = wavelet_size

        # Get extra paramters
        if params is None:
            params = {}

        p_drop = params.get('dropout_rate', .05)
        k_size = params.get('kernel_size', 3)
        padding = k_size // 2
        downsampling_factor = params.get('downsampling_factor', 3)

        self.p_drop = p_drop

        n_in_channels = 2


        # ENCODE
        se_params = dict(factor=downsampling_factor, kernel_size=k_size, padding=padding)
        modules = []
        if one_ms:
            modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=128, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        else:
            modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        self.encoder = Sequential(modules)


        # LATENT
        self.fc_mu = nn.Linear(256, 128)
        self.fc_logvar = nn.Linear(256, 128)


        # DECODE
        modules = []
        if one_ms:
            modules += [LinearLrelu(in_features=128,out_features=512)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [nn.Linear(1024, wavelet_size)]
        else:
            modules += [LinearLrelu(in_features=128,out_features=512)]
            modules += [nn.Dropout(p=p_drop, inplace = True)]
            modules += [nn.Linear(512, wavelet_size)]
        self.wavelet_decoder = Sequential(modules)


        # LEARN AMPLITUDE
        self.scaling = ScalingOperator()


    def encode(self, seismic: Tensor, reflecticvity: Tensor) -> Tuple[Tensor]:
        """
        Concatenates inputs in  channel dims and
        encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x L]
        :return: (Tensor) List of latent codes
        """
        # cat
        x = torch.cat((seismic, reflecticvity), dim=1)

        # encode
        x = self.encoder(x)

        # global average pooling (gap)
        x = torch.mean(x, 2, keepdim=False)

        # drop
        x = F.dropout(x, p=self.p_drop, training=self.training)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var


    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        mu : (Tensor) Mean of the latent Gaussian
        log_var : (Tensor) Logarithm of the varinace of the latent Gaussian
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


    def decode(self, z: Tensor) -> Tensor:
        w = self.wavelet_decoder(z)
        w = self.scaling(w)
        return w


    def sample(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        wavelet, _, _ = self.forward(seismic, reflectivity)
        return wavelet

    def sample_n_times(self, seismic: Tensor, reflectivity: Tensor, n: int) -> List[Tensor]:
        wavelets_distribution = []
        for _ in range(n):
            wavelets_distribution.append(self.sample(seismic, reflectivity))
        return wavelets_distribution


    def compute_expected_wavelet(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        mu, _ = self.encode(seismic, reflectivity)
        wavelet = self.decode(mu)
        return wavelet


    def forward(self, seismic: Tensor, reflectivity: Tensor) -> Tuple[Tensor]:
        mu, log_var = self.encode(seismic, reflectivity)
        z = self.reparameterize(mu, log_var)
        wavelet = self.decode(z)
        return wavelet, mu, log_var

###################################
# Utils classes
###################################

class Sequential(nn.Module):
    def __init__(self, modules: List[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x



class ScalingOperator(nn.Module):
  def __init__(self):
    super().__init__()

    self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)

  def forward(self, x: Tensor) -> Tensor:
    x *= self.scale
    return x




