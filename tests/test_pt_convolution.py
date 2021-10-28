import torch
import numpy as np

from wtie.processing.spectral import pt_convolution


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_pt_conv():
    batch_size = 12

    wlts = np.random.normal(size=(batch_size, 1, 42)).astype(np.float32)
    refs = np.random.normal(size=(batch_size, 1, 101)).astype(np.float32)

    seismics = pt_convolution(torch.from_numpy(wlts).to(device),
                              torch.from_numpy(refs).to(device))
    # correct shape?
    assert seismics.shape[0] == batch_size
    assert seismics.shape[1] == 1

    # correct computation?
    # test for first sample of the batch
    seismic_0_np = np.convolve(wlts[0,0,:], refs[0,0,:], mode='valid')
    assert np.allclose(seismic_0_np, seismics[0,0,:].cpu().numpy(), atol=1e-03)





