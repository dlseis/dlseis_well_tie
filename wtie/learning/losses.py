"""Loss classes"""

import torch
import math

from wtie.utils.types_ import Tensor, Dict
from wtie.processing.spectral import pt_convolution


class AbstractLoss(object):
    key_names = None
    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError("Losses subcalsses must implement `key_names` attribute")

        if 'total' not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backprop.")




class ReconstructionLoss(AbstractLoss):
    key_names = ('total', 'wavelet', 'seismic')

    def __init__(self, parameters: dict, is_validation_error: bool=True):
        self.key_names = ReconstructionLoss.key_names
        super().__init__()

        self.wavelet_loss_type = parameters['wavelet_loss_type']
        self.seismic_loss_type = parameters['seismic_loss_type']
        self.beta = parameters['beta']

        self.is_validation_error = is_validation_error


    def __call__(self,
                 data_batch: dict,
                 predicted_wavelet: Tensor,
                 ) -> Dict[str, float]:
        #recons_loss = F.mse_loss(inp, out, reduction='sum')
        # torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0)

        #---------------
        # Wavelet error
        #---------------
        label_wavelet = data_batch['wavelet']
        if self.wavelet_loss_type == 'mse':
            loss_wavelet = \
                torch.mean(torch.sum((label_wavelet - predicted_wavelet).pow(2),dim=1),dim=0)
        elif self.wavelet_loss_type == 'mae':
            loss_wavelet = \
                torch.mean(torch.sum(torch.abs(label_wavelet - predicted_wavelet),dim=1),dim=0)
        else:
            raise ValueError("Wrong wavelet loss type %s" % str(self.wavelet_loss_type))

        #---------------
        # Seismic error
        #---------------
        beta = self.beta
        cond = (beta is not None) and beta > 0.
        if cond:
            # add channel = 1
            true_seismic = pt_convolution(torch.unsqueeze(label_wavelet, 1),
                                          data_batch['reflectivity'])
            pred_seismic = pt_convolution(torch.unsqueeze(predicted_wavelet, 1),
                                          data_batch['reflectivity'])
            # shape [batch, 1, N]
            if self.seismic_loss_type == 'mse':
                loss_seismic = \
                    torch.mean(torch.sum((true_seismic - pred_seismic).pow(2),dim=2),dim=(0,1))
            elif self.seismic_loss_type == 'mae':
                loss_seismic = \
                    torch.mean(torch.sum(torch.abs(true_seismic - pred_seismic),dim=2),dim=(0,1))
            else:
                raise ValueError("Wrong seismic loss type %s" % str(self.seismic_loss_type))



        #-----------------
        # Total error
        #-----------------
        if cond:
            total_loss = (1 - beta) * loss_wavelet + beta * loss_seismic
            loss = {'total':total_loss, 'wavelet':loss_wavelet, 'seismic':loss_seismic}
        else:
            loss = {'total':loss_wavelet,
                    'wavelet':loss_wavelet,
                    'seismic':torch.zeros((1,), dtype=torch.float32, requires_grad=False)
                    }

        #------------------
        # Validation error (used for hyper-parameter search, not for training)
        #------------------
        if self.is_validation_error:
            #validation_error = torch.sum(torch.abs(label_wavelet - predicted_wavelet), dim=1)
            validation_error = torch.sum((label_wavelet - predicted_wavelet).pow(2), dim=1)
            loss['validation_error_mean'] = torch.mean(validation_error)
            loss['validation_error_std'] = torch.std(validation_error)

        return loss





class _CenteredUnitGaussianLoss(AbstractLoss):
    """Assumes a zero centered unit Gaussian distribution."""
    key_names = ('variational', 'total')

    def __init__(self, parameters: dict=None):
        self.key_names = _CenteredUnitGaussianLoss.key_names
        super().__init__()

    def __call__(self, mu: Tensor, log_var: Tensor) -> Dict[str, float]:
        batch_kld_loss = _batch_kl_div_with_unit_gaussian(mu, log_var)

        kld_loss = torch.mean(batch_kld_loss, dim=0)

        return {'variational':kld_loss, 'total':kld_loss}




class VariationalLoss(AbstractLoss):
    key_names = ('total', 'wavelet', 'seismic', 'variational')

    def __init__(self, parameters: dict):
        self.key_names = VariationalLoss.key_names
        super().__init__()

        self.reconstruction = ReconstructionLoss(parameters,
                                                 is_validation_error=False)
        self.kl_div = _CenteredUnitGaussianLoss()

        # weighting between reconstruction and variation
        self._alpha = parameters['alpha_init']

    def __call__(self,
                 data_batch: dict,
                 predicted_wavelet: Tensor,
                 mu: Tensor,
                 log_var: Tensor
                 ) -> Dict[str, float]:

        loss_dict = {}
        alpha = self.alpha

        # compute losses
        reconstruction_dict = self.reconstruction(data_batch, predicted_wavelet)
        kl_dict = self.kl_div(mu, log_var)

        # total loss
        total_loss = (1 - alpha) * reconstruction_dict['total'] + alpha * kl_dict['total']

        # place in dict
        for key, value in reconstruction_dict.items():
            if key != 'total':
                loss_dict[key] = value

        for key, value in kl_dict.items():
            if key != 'total':
                loss_dict[key] = value

        loss_dict['total'] = total_loss


        #------------------
        # Validation error (used for hyper-parameter search, not for training)
        #------------------

        # api: https://ax.dev/docs/trial-evaluation.html
        # Dict[Tuple[mean: float, std: float]]
        validation_error = {}

        # reconstruction
        reconstrcution_validation_error = \
            torch.sum(torch.abs(data_batch['wavelet'] - predicted_wavelet), dim=1)
        reconstrcution_validation_error_mean = torch.mean(reconstrcution_validation_error)
        reconstrcution_validation_error_std = torch.std(reconstrcution_validation_error)
        validation_error['reconstruction_error'] = \
            (reconstrcution_validation_error_mean, reconstrcution_validation_error_std)

        # variation
        batch_kld_loss = _batch_kl_div_with_unit_gaussian(mu, log_var) #NaN problems?
        #batch_kld_loss = _dummy_mu_log_var_batch_error(mu, log_var)
        kld_loss_mean = torch.mean(batch_kld_loss)
        kld_loss_std = torch.std(batch_kld_loss)
        validation_error['variation_error'] = \
            (kld_loss_mean, kld_loss_std)


        loss_dict['validation_error'] = validation_error


        return loss_dict

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value






def _batch_kl_div_with_unit_gaussian(mu: Tensor, log_var: Tensor) -> Tensor:
        return (-0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1))


def OLD_tmp_mu_log_var_batch_error(mu: Tensor, log_var: Tensor) -> Tensor:
    # So far there are NaNs problems with kl_div...
    sigma = torch.exp(0.5*log_var) # positve

    # mu to 0 and sigma to 1

    # l1
    #batch_mu_error = torch.sum(torch.abs(mu), dim=1)
    #batch_sigma_error = torch.sum(torch.abs(sigma - torch.ones_like(sigma)), dim=1)

    # l2
    batch_mu_error = 100 * torch.sum(mu.pow(2), dim=1)
    batch_sigma_error = 100 * torch.sum((sigma - torch.ones_like(sigma)).pow(2), dim=1)

    return batch_mu_error + batch_sigma_error









