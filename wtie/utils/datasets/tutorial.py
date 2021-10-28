"""Import utils for the tutorial data.

NOTE:
The data used in this tutorial comes from the well 159_19A of Equinor's open
Volve dataset, and from the Poseidon dataset.

Links:
https://www.equinor.com/en/what-we-do/digitalisation-in-our-dna/volve-field-data-village-download.html
https://terranubis.com/datainfo/NW-Shelf-Australia-Poseidon-3D
"""

import numpy as np
import pandas as pd
import segyio
import lasio

from pathlib import Path


from wtie.processing import grid
from wtie.processing.logs import smoothly_interpolate_nans, interpolate_nans
from wtie.learning.model import VariationalNetwork, VariationalEvaluator
from wtie.modeling.modeling import ConvModeler
from wtie.utils.datasets.utils import InputSet


def get_modeling_tool():
    return ConvModeler()


def load_wavelet_extractor(training_parameters: dict,
                           networ_state_dict: Path
                           ) -> VariationalEvaluator:
    # instanciate neural network
    output_size = training_parameters['synthetic_dataset']['wavelet']['wavelet_size']
    net = VariationalNetwork(output_size,
                             training_parameters['network']['network_kwargs'])

    # instanciate evaluator object
    expected_sampling_rate = training_parameters['synthetic_dataset']['dt']
    evaluator = VariationalEvaluator(network=net,
                                     expected_sampling=expected_sampling_rate,
                                     state_dict=networ_state_dict)

    return evaluator


#########################
# Poseidon
#########################


def load_poseidon_data(folder: Path, well: str = 'torosa1') -> InputSet:
    folder = Path(folder) / 'Poseidon' / well

    if well == 'torosa1':
        logset_md = _import_torosa1_logs(folder)
        seismic = _import_torosa1_seismic(folder)
        wellpath = _import_torosa1_well_path(folder)
        table = _import_torosa1_time_depth_table(folder)
    elif well == 'boreas1':
        logset_md = _import_boreas1_logs(folder)
        seismic = _import_boreas1_seismic(folder)
        wellpath = _import_boreas1_well_path(folder)
        table = _import_boreas1_time_depth_table(folder)
    else:
        raise NotImplementedError(f"{well} well is not available.")
    return InputSet(logset_md, seismic, wellpath, table)


########## Torosa ############

def _import_torosa1_logs(folder: str) -> grid.LogSet:
    file_path = Path(folder) / 'Torosa1Decim.LAS'

    # Read file
    las_logs = lasio.read(file_path)

    # valid range
    start_burn = 6050
    end_burn = 60

    # Vp
    sonic = las_logs.df().loc[:, 'BATC'].values[start_burn:-end_burn]  # us/ft
    sonic = 1 / sonic  # ft/us
    sonic *= 1e6  # ft/s
    sonic /= 3.2808  # m/s
    md = las_logs.df().loc[:, 'BATC'].index.values[start_burn:-end_burn]
    Vp = grid.Log(sonic, md, 'md', name='Vp', allow_nan=False)

    # Rho
    _rho = las_logs.df().RHOZ.values[start_burn:-end_burn]
    md = las_logs.df().RHOZ.index.values[start_burn:-end_burn]
    Rho = grid.Log(_rho, md, 'md', name='Rho', unit='g/cm3', allow_nan=False)

    # Shear
    shear = las_logs.df().loc[:, 'DTS'].values[start_burn:-end_burn]
    shear = 1 / shear  # ft/us
    shear *= 1e6  # ft/s
    shear /= 3.2808  # m/s
    shear = interpolate_nans(shear)
    md = las_logs.df().loc[:, 'DTS'].index.values[start_burn:-end_burn]
    Vs = grid.Log(shear, md, 'md', name='Vs', allow_nan=False)

    # GR
    gr = las_logs.df().GR.values[start_burn:-end_burn]
    md = las_logs.df().GR.index.values[start_burn:-end_burn]
    GR = grid.Log(gr, md, 'md', name='Gamma Ray', allow_nan=True)

    # TNP
    htnp = las_logs.df().HTNP.values[start_burn:-end_burn]
    md = las_logs.df().HTNP.index.values[start_burn:-end_burn]
    HNTP = grid.Log(
        htnp, md, 'md', name='Thermal Neutron Porosity', allow_nan=True)

    return grid.LogSet(logs={'Vp': Vp, 'Rho': Rho, 'Vs': Vs, 'GR': GR, 'TNP': HNTP})


def _import_torosa1_well_path(folder: str) -> grid.WellPath:
    file_path = Path(folder) / 'Torosa1_dev.txt'

    _wp = pd.read_csv(file_path, header=1, delimiter=r"\s+",
                      names=('Depth', 'Inclination', 'Azimuth'),
                      usecols=range(1, 4))
    # datum
    kb = 22.9  # meters

    _md = np.concatenate((np.zeros((1,)), _wp.loc[:, 'Depth'].values))
    _dev = np.concatenate(
        (np.zeros((1,)), _wp.loc[:, 'Inclination'].values[:-1]))

    _tvd = grid.WellPath.get_tvdkb_from_inclination(_md, _dev)
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    # 0,0
    #_md = np.concatenate((np.zeros((1,)), _md))
    #_tvd = np.concatenate((np.zeros((1,)), _tvd))

    return grid.WellPath(md=_md, tvdss=_tvd, kb=kb)


def _import_torosa1_time_depth_table(folder: str) -> grid.TimeDepthTable:
    file_path = Path(folder) / 'Torosa1_TZV_DEPTH_CS_CALIBRATED_LOGS.LAS'

    df = lasio.read(file_path).df()

    _twt = df.loc[:, 'TIME'].values / 1e3  # ms to s
    _tvd = df.loc[:, 'TVD'].values  # from MSL

    # remove nans
    good_idx = np.where(~ np.isnan(_twt))[0]
    _twt = _twt[good_idx]
    _tvd = _tvd[good_idx]

    # TODO: check dataum...
    # kb = 22.9 # meters
    #_tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvd)


def _import_torosa1_seismic(folder: str) -> grid.Seismic:
    file_path = Path(folder) / 'Torosa1_seismic_alongwell_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
    return grid.Seismic(_seis, _twt, 'twt', name='Torosa1 seismic')


########## Boreas ############
def _import_boreas1_logs(folder: str) -> grid.LogSet:
    file_path = Path(folder) / 'Boreas1Decim.las'

    # Read file
    las_logs = lasio.read(file_path)

    # valid range
    start_burn = 7650
    end_burn = 70

    # Vp
    sonic = las_logs.df().DTCO.values[start_burn:-end_burn]  # us/ft
    sonic = 1 / sonic  # ft/us
    sonic *= 1e6  # ft/s
    sonic /= 3.2808  # m/s
    md = las_logs.df().DTCO.index.values[start_burn:-end_burn]
    Vp = grid.Log(sonic, md, 'md', name='Vp', allow_nan=False)

    # Rho
    _rho = las_logs.df().RHOB.values[start_burn:-end_burn]
    rho2 = smoothly_interpolate_nans(_rho,
                                     despike_params=dict(
                                         median_size=41, threshold=1.0),
                                     smooth_params=dict(std=4.0)
                                     )
    md = las_logs.df().RHOB.index.values[start_burn:-end_burn]
    Rho = grid.Log(rho2, md, 'md', name='Rho', unit='g/cm3', allow_nan=False)

    # Shear
    shear = las_logs.df().DTSM.values[start_burn:-end_burn]  # us/ft
    shear = 1 / shear  # ft/us
    shear *= 1e6  # ft/s
    shear /= 3.2808  # m/s
    md = las_logs.df().DTSM.index.values[start_burn:-end_burn]
    Vs = grid.Log(shear, md, 'md', name='Vs', allow_nan=True)

    # Porosity
    tnp = las_logs.df().TNPH.values[start_burn:-end_burn]  # us/ft
    md = las_logs.df().TNPH.index.values[start_burn:-end_burn]
    tnP = grid.Log(tnp, md, 'md', name='Term Neur Porosity', allow_nan=True)

    # Cal
    hd = las_logs.df().HDAR.values[start_burn:-end_burn]  # us/ft
    md = las_logs.df().HDAR.index.values[start_burn:-end_burn]
    HD = grid.Log(hd, md, 'md', name='Hole diameter', allow_nan=True)

    return grid.LogSet(logs={'Vp': Vp, 'Rho': Rho, 'Vs': Vs, 'Cali': HD, 'Porosity': tnP})


def _import_boreas1_well_path(folder: str) -> grid.WellPath:
    file_path = Path(folder) / 'Boreas1_dev.txt'

    _wp = pd.read_csv(file_path, header=1, delimiter=r"\s+", usecols=range(1, 7),
                      names=('Depth', 'Dev', 'Azimuth', 'Depth2', 'Dev2', 'Azimuth2'))

    kb = 0  # TODO, UNKNOWN... # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(
        _wp.loc[:, 'Depth'].values,
        _wp.loc[:, 'Dev'].values[:-1]
    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:, 'Depth'].values, tvdss=_tvd, kb=kb)


def _import_boreas1_time_depth_table(folder: str) -> grid.TimeDepthTable:
    file_path = Path(folder) / 'Boreas1_vel.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0, 1],
                      names=('md', 'tvdss', 'owt', 'md2', 'tvdss2', 'owt2'),
                      usecols=range(1, 7))

    _twt = _td.loc[:, 'owt'].values * 2  # owt to twt
    _tvdss = _td.loc[:, 'tvdss'].values

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvdss)


def _import_boreas1_seismic(folder: str) -> grid.Seismic:
    file_path = Path(folder) / 'Boreas1_seismic_alongwell_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
    return grid.Seismic(_seis, _twt, 'twt', name='Boreas1 seismic')


#########################
# Volve 159-19A
#########################

def load_volve_data(folder: Path, prestack: bool = True) -> InputSet:
    folder = Path(folder) / 'Volve'
    # data
    logset_md = _import_volve_logs(folder)
    seismic = _import_volve_seismic(folder, prestack=prestack)
    wellpath = _import_volve_well_path(folder)
    table = _import_volve_time_depth_table(folder)
    return InputSet(logset_md, seismic, wellpath, table)


def _import_volve_logs(folder: str) -> grid.LogSet:
    file_path = Path(folder) / 'volve_159-19A_LFP.las'

    # Read file
    las_logs = lasio.read(file_path)
    las_logs = las_logs.df()

    # Select some logs, there are more, we only load the following.
    # Must at least contain the keys 'Vp' for acoustic velocity
    # and 'Rho' for the bulk density. 'Vs', for shear velocity, must also
    # be imported if one whishes to perform a prestack well-tie.
    # Other logs are optional.
    log_dict = {}

    log_dict['Vp'] = grid.Log(las_logs.LFP_VP.values,
                              las_logs.LFP_VP.index, 'md', name='Vp')
    log_dict['Vs'] = grid.Log(las_logs.LFP_VS.values,
                              las_logs.LFP_VS.index, 'md', name='Vs')

    # Density contains some NaNs, I fill them with linear interpolation.
    log_dict['Rho'] = grid.Log(interpolate_nans(las_logs.LFP_RHOB.values),
                               las_logs.LFP_RHOB.index, 'md', name='Rho')

    log_dict['GR'] = grid.Log(interpolate_nans(
        las_logs.LFP_GR.values), las_logs.LFP_VP.index, 'md')
    log_dict['Cali'] = grid.Log(
        las_logs.LFP_CALI.values, las_logs.LFP_VP.index, 'md')

    return grid.LogSet(log_dict)


def _import_volve_seismic(folder: str, prestack: bool = False
                          ) -> grid.seismic_t:

    if prestack:
        file_path = Path(folder) / 'volve_15_9_19A_gather.sgy'

        with segyio.open(file_path, 'r') as f:
            _twt = f.samples / 1000  # two-way-time in seconds
            _seis = np.squeeze(segyio.tools.cube(f))  # 2D (angles, samples)
            _angles = f.offsets  # in degrees

        seismic = []
        for i, theta in enumerate(_angles):
            seismic.append(grid.Seismic(_seis[i, :], _twt, 'twt', theta=theta))
        seismic = grid.PreStackSeismic(seismic, name='Real gather')
        # # stacking from 2° to 10°
        # _seis = np.mean(_seis[2:10,:], axis=0)
        # seismic = grid.Seismic(_seis, _twt, 'twt', name='Real seismic')

    else:
        file_path = Path(folder) / 'volve_5_9_19A_Seismic_0_0.sgy'
        with segyio.open(file_path, 'r') as f:
            _twt = f.samples / 1000  # two-way-time in seconds
            _seis = np.squeeze(segyio.tools.cube(f))  # 1D
        seismic = grid.Seismic(_seis, _twt, 'twt', name='Real seismic')

    return seismic


def _import_volve_well_path(folder: str) -> grid.WellPath:
    file_path = Path(folder) / 'volve_path_15_9-19_A.txt'

    _wp = pd.read_csv(file_path, header=None, delimiter=r"\s+",
                      names=('MD (kb) [m]', 'Inclination', 'Azimuth'))

    kb = 25  # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(
        _wp.loc[:, 'MD (kb) [m]'].values,
        _wp.loc[:, 'Inclination'].values[:-1]
    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:, 'MD (kb) [m]'].values, tvdss=_tvd, kb=kb)


def _import_volve_time_depth_table(folder: str) -> grid.TimeDepthTable:
    file_path = Path(folder) / 'volve_checkshot_15_9_19A.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0],
                      names=('Curve Name', 'TVDBTDD', 'TVDKB', 'TVDSS', 'TWT'))

    _twt = _td.loc[:, 'TWT'].values / 1000
    _tvdss = np.abs(_td.loc[:, 'TVDSS'].values)

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvdss)
