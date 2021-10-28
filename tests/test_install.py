import torch
import warnings


def is_cuda():
    if not torch.cuda.is_available():
        mes = "cuda was not detected (cannot use GPU), please check pytorch and cudatoolkit install."
        warnings.warn(UserWarning(mes))
    return 1


def test_is_cuda():
    assert is_cuda() == 1
