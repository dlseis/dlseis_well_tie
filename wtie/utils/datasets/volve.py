"""Utils to import Volve data."""

import numpy as np
import pandas as pd
import segyio
import lasio

from pathlib import Path

from wtie.processing import grid
from wtie.processing.logs import interpolate_nans





def import_logs(file_path: str) -> grid.LogSet:
    file_path = Path(file_path)
    assert file_path.name == '159-19A_LFP.las'

    # Read file
    las_logs = lasio.read(file_path)
    las_logs = las_logs.df()

    # Select some logs
    log_dict = {}



    log_dict['Vp'] = grid.Log(las_logs.LFP_VP.values, las_logs.LFP_VP.index, 'md', name='Vp')
    log_dict['Vs'] = grid.Log(las_logs.LFP_VS.values, las_logs.LFP_VS.index, 'md', name='Vs')
    log_dict['Rho'] = grid.Log(interpolate_nans(las_logs.LFP_RHOB.values),
                    las_logs.LFP_RHOB.index, 'md', name='Rho')

    log_dict['GR'] = grid.Log(interpolate_nans(las_logs.LFP_GR.values), las_logs.LFP_VP.index, 'md')
    log_dict['Cali'] = grid.Log(las_logs.LFP_CALI.values, las_logs.LFP_VP.index, 'md')

    return grid.LogSet(log_dict)


def import_well_path(file_path: str) -> grid.WellPath:
    file_path = Path(file_path)
    assert file_path.name == 'path_15_9-19_A.txt'

    _wp = pd.read_csv(file_path, header=None, delimiter=r"\s+",
                      names=('MD (kb) [m]', 'Inclination', 'Azimuth'))

    kb = 25 # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(\
                                            _wp.loc[:,'MD (kb) [m]'].values,
                                            _wp.loc[:,'Inclination'].values[:-1]
                                                    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:,'MD (kb) [m]'].values, tvdss=_tvd, kb=kb)


def import_time_depth_table(file_path: str) -> grid.TimeDepthTable:
    file_path = Path(file_path)
    assert file_path.name == 'checkshot_15_9_19A.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0],
                  names=('Curve Name', 'TVDBTDD', 'TVDKB', 'TVDSS', 'TWT'))

    _twt = _td.loc[:,'TWT'].values / 1000
    _tvdss = np.abs(_td.loc[:,'TVDSS'].values)

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvdss)



def import_seismic(file_path: str) -> grid.Seismic:
    file_path = Path(file_path)
    assert file_path.name == '5_9_19A_Seismic_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))

    return grid.Seismic(_seis, _twt, 'twt', name='Real seismic')


def import_prestack_seismic(file_path: str) -> grid.PreStackSeismic:
    file_path = Path(file_path)
    assert file_path.name == '5_9_19A_gather.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
        _angles = f.offsets

    seismic = []
    for i, theta in enumerate(_angles):
        seismic.append(grid.Seismic(_seis[i,:], _twt, 'twt', theta=theta))

    return grid.PreStackSeismic(seismic, name='Real gather')


def import_psp_reflectivity(file_path: str) -> grid.Reflectivity:
    file_path = Path(file_path)
    assert file_path.name == '5_9_19A_Refl_10146_10146.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))

    return grid.Reflectivity(_seis, _twt, 'twt', name='psp reflectivity')
