import yaml

with open('../experiments/parameters.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)


def test_params():
    assert type(params['network']['wavelet_loss_type']) is str
    assert type(params['synthetic_dataset']) is dict