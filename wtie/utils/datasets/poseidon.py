import segyio
import lasio

import numpy as np
import pandas as pd

from pathlib import Path


from wtie.processing import grid
from wtie.processing.logs import (smoothly_interpolate_nans, interpolate_nans)

_DEFAULT_WELL: str = 'boreas1'
_error_msg = lambda name : ("Import function for well %s has not been implemented." % name)


def import_logs(file_path: str, well: str=_DEFAULT_WELL):
    #well = well.lower()
    if well == 'boreas1':
        return _import_boreas1_logs(file_path)
    elif well == 'pharos1':
        return _import_pharos1_logs(file_path)
    elif well == 'torosa1':
        return _import_torosa1_logs(file_path)
    else:
        raise NotImplementedError(_error_msg(well))


def import_well_path(file_path: str, well: str=_DEFAULT_WELL):
    #well = well.lower()
    if well == 'boreas1':
        return _import_boreas1_well_path(file_path)
    elif well == 'pharos1':
        return _import_pharos1_well_path(file_path)
    elif well == 'torosa1':
        return _import_torosa1_well_path(file_path)
    else:
        raise NotImplementedError(_error_msg(well))


def import_time_depth_table(file_path: str, well: str=_DEFAULT_WELL):
    #well = well.lower()
    if well == 'boreas1':
        return _import_boreas1_time_depth_table(file_path)
    elif well == 'pharos1':
        return _import_pharos1_time_depth_table(file_path)
    elif well == 'torosa1':
        return _import_torosa1_time_depth_table(file_path)
    else:
        raise NotImplementedError(_error_msg(well))


def import_seismic(file_path: str, well: str=_DEFAULT_WELL):
    #well = well.lower()
    if well == 'boreas1':
        return _import_boreas1_seismic(file_path)
    elif well == 'pharos1':
        return _import_pharos1_seismic(file_path)
    elif well == 'torosa1':
        return _import_torosa1_seismic(file_path)
    else:
        raise NotImplementedError(_error_msg(well))


########################################
# TOROSA1
########################################
def _import_torosa1_logs(file_path: str) -> grid.LogSet:
    file_path = Path(file_path)
    assert file_path.name == 'Torosa1Decim.LAS'

    # Read file
    las_logs = lasio.read(file_path)

    # valid range
    start_burn = 6050
    end_burn = 60

    # Vp
    sonic = las_logs.df().loc[:,'BATC'].values[start_burn:-end_burn] # us/ft
    sonic = 1 / sonic # ft/us
    sonic *= 1e6 # ft/s
    sonic /= 3.2808 # m/s
    md = las_logs.df().loc[:,'BATC'].index.values[start_burn:-end_burn]
    Vp = grid.Log(sonic, md, 'md', name='Vp', allow_nan=False)

    # Rho
    _rho = las_logs.df().RHOZ.values[start_burn:-end_burn]
    md = las_logs.df().RHOZ.index.values[start_burn:-end_burn]
    Rho = grid.Log(_rho, md, 'md', name='Rho', unit='g/cm3', allow_nan=False)

    # Shear
    shear = las_logs.df().loc[:,'DTS'].values[start_burn:-end_burn]
    shear = 1 / shear # ft/us
    shear *= 1e6 # ft/s
    shear /= 3.2808 # m/s
    shear = interpolate_nans(shear)
    md = las_logs.df().loc[:,'DTS'].index.values[start_burn:-end_burn]
    Vs = grid.Log(shear, md, 'md', name='Vs', allow_nan=False)

    # GR
    gr = las_logs.df().GR.values[start_burn:-end_burn]
    md = las_logs.df().GR.index.values[start_burn:-end_burn]
    GR = grid.Log(gr, md, 'md', name='Gamma Ray', allow_nan=True)

    # TNP
    htnp = las_logs.df().HTNP.values[start_burn:-end_burn]
    md = las_logs.df().HTNP.index.values[start_burn:-end_burn]
    HNTP = grid.Log(htnp, md, 'md', name='Thermal Neutron Porosity', allow_nan=True)



    return grid.LogSet(logs={'Vp':Vp, 'Rho':Rho, 'Vs':Vs, 'GR':GR, 'TNP':HNTP})


def _import_torosa1_well_path(file_path: str) -> grid.WellPath:
    file_path = Path(file_path)
    assert file_path.name == 'Torosa1_dev.txt'

    _wp = pd.read_csv(file_path, header=1, delimiter=r"\s+",
                      names=('Depth', 'Inclination', 'Azimuth'),
                      usecols=range(1,4))
    # datum
    kb = 22.9 # meters

    _md = np.concatenate((np.zeros((1,)), _wp.loc[:,'Depth'].values))
    _dev = np.concatenate((np.zeros((1,)), _wp.loc[:,'Inclination'].values[:-1]))

    _tvd = grid.WellPath.get_tvdkb_from_inclination(_md, _dev)
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)



    # 0,0
    #_md = np.concatenate((np.zeros((1,)), _md))
    #_tvd = np.concatenate((np.zeros((1,)), _tvd))

    return grid.WellPath(md=_md, tvdss=_tvd, kb=kb)


def _import_torosa1_time_depth_table(file_path: str) -> grid.TimeDepthTable:
    file_path = Path(file_path)
    assert file_path.name == 'Torosa1_TZV_DEPTH_CS_CALIBRATED_LOGS.LAS'

    df = lasio.read(file_path).df()

    _twt = df.loc[:,'TIME'].values / 1e3 # ms to s
    _tvd = df.loc[:,'TVD'].values # from MSL

    # remove nans
    good_idx = np.where(~ np.isnan(_twt))[0]
    _twt = _twt[good_idx]
    _tvd = _tvd[good_idx]

    #TODO: check dataum...
    #kb = 22.9 # meters
    #_tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvd)


def _import_torosa1_seismic(file_path: str) -> grid.Seismic:
    file_path = Path(file_path)
    assert file_path.name == 'Torosa1_seismic_alongwell_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
    return grid.Seismic(_seis, _twt, 'twt', name='Torosa1 seismic')
########################################
# PHAROS1
########################################
def _import_pharos1_logs(file_path: str) -> grid.LogSet:
    file_path = Path(file_path)
    assert file_path.name == 'COP_Pharos-1_logs_depth.las'

    # Read file
    las_logs = lasio.read(file_path)

    # valid range
    start_burn = 26350
    end_burn = 40

    # Vp
    sonic = las_logs.df().loc[:,'DTCO:1'].values[start_burn:-end_burn] # us/ft
    sonic = 1 / sonic # ft/us
    sonic *= 1e6 # ft/s
    sonic /= 3.2808 # m/s
    md = las_logs.df().loc[:,'DTCO:1'].index.values[start_burn:-end_burn]
    Vp = grid.Log(sonic, md, 'md', name='Vp', allow_nan=False)

    # Rho
    _rho = las_logs.df().RHOB.values[start_burn:-end_burn]
    md = las_logs.df().RHOB.index.values[start_burn:-end_burn]
    Rho = grid.Log(_rho, md, 'md', name='Rho', unit='g/cm3', allow_nan=False)

    # Shear
    shear = las_logs.df().loc[:,'DTTS'].values[start_burn:-end_burn]
    shear = 1 / shear # ft/us
    shear *= 1e6 # ft/s
    shear /= 3.2808 # m/s
    shear = interpolate_nans(shear)
    md = las_logs.df().loc[:,'DTTS'].index.values[start_burn:-end_burn]
    Vs = grid.Log(shear, md, 'md', name='Vs', allow_nan=True)

    # GR
    gr = las_logs.df().GR.values[start_burn:-end_burn]
    md = las_logs.df().GR.index.values[start_burn:-end_burn]
    GR = grid.Log(gr, md, 'md', name='Gamma Ray', allow_nan=False)


    return grid.LogSet(logs={'Vp':Vp, 'Rho':Rho, 'Vs':Vs, 'GR':GR})


def _import_pharos1_well_path(file_path: str) -> grid.WellPath:
    file_path = Path(file_path)
    assert file_path.name == 'Pharos1_dev_raw.txt'

    _wp = pd.read_csv(file_path, header=0, delimiter=r"\s+",
                      names=('Depth', 'Inclination', 'Azimuth'))

    kb = 22.10 # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(\
                                            _wp.loc[:,'Depth'].values,
                                            _wp.loc[:,'Inclination'].values[:-1]
                                                    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:,'Depth'].values, tvdss=_tvd, kb=kb)


def _import_pharos1_time_depth_table(file_path: str) -> grid.TimeDepthTable:
    file_path = Path(file_path)
    assert file_path.name == 'Pharos1_vel_raw.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0],
                  names=('md', 'tvd', 'twt'))

    _twt = 2 * (_td.loc[:,'twt'].values / 1e3) # owt to twt, ms to s
    _tvd = _td.loc[:,'tvd'].values # from SRD (seismic reference datum)

    #TODO: check dataum...
    #kb = 22.10 # meters
    #_tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvd)


def _import_pharos1_seismic(file_path: str) -> grid.Seismic:
    file_path = Path(file_path)
    assert file_path.name == 'Pharos1_seismic_alongwell2_0_0.sgy'#'Pharos1_seismic_alongwell_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
    _seis *= -1.0
    return grid.Seismic(_seis, _twt, 'twt', name='Pharos1 seismic')

########################################
# BOREAS1
########################################
def _import_boreas1_logs(file_path: str) -> grid.LogSet:
    file_path = Path(file_path)
    assert file_path.name == 'Boreas1Decim.las'

    # Read file
    las_logs = lasio.read(file_path)

    # valid range
    start_burn = 7650
    end_burn = 70

    # Vp
    sonic = las_logs.df().DTCO.values[start_burn:-end_burn] # us/ft
    sonic = 1 / sonic # ft/us
    sonic *= 1e6 # ft/s
    sonic /= 3.2808 # m/s
    md = las_logs.df().DTCO.index.values[start_burn:-end_burn]
    Vp = grid.Log(sonic, md, 'md', name='Vp', allow_nan=False)

    # Rho
    _rho = las_logs.df().RHOB.values[start_burn:-end_burn]
    rho2 = smoothly_interpolate_nans(_rho,
                                 despike_params=dict(median_size=41, threshold=1.0),
                                 smooth_params=dict(std=4.0)
                                )
    md = las_logs.df().RHOB.index.values[start_burn:-end_burn]
    Rho = grid.Log(rho2, md, 'md', name='Rho', unit='g/cm3', allow_nan=False)

    # Shear
    shear = las_logs.df().DTSM.values[start_burn:-end_burn] # us/ft
    shear = 1 / shear # ft/us
    shear *= 1e6 # ft/s
    shear /= 3.2808 # m/s
    md = las_logs.df().DTSM.index.values[start_burn:-end_burn]
    Vs = grid.Log(shear, md, 'md', name='Vs', allow_nan=True)

    # Porosity
    tnp = las_logs.df().TNPH.values[start_burn:-end_burn] # us/ft
    md = las_logs.df().TNPH.index.values[start_burn:-end_burn]
    tnP = grid.Log(tnp, md, 'md', name='Term Neur Porosity', allow_nan=True)

    # Cal
    hd = las_logs.df().HDAR.values[start_burn:-end_burn] # us/ft
    md = las_logs.df().HDAR.index.values[start_burn:-end_burn]
    HD = grid.Log(hd, md, 'md', name='Hole diameter', allow_nan=True)

    return grid.LogSet(logs={'Vp':Vp, 'Rho':Rho, 'Vs':Vs, 'Cali':HD, 'Porosity':tnP})


def _import_boreas1_well_path(file_path: str) -> grid.WellPath:
    file_path = Path(file_path)
    assert file_path.name == 'Boreas1_dev.txt'

    _wp = pd.read_csv(file_path, header=1, delimiter=r"\s+", usecols=range(1,7),
                  names=('Depth', 'Dev', 'Azimuth', 'Depth2', 'Dev2', 'Azimuth2'))

    kb = 0 # TODO, UNKNOWN... # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(\
                                            _wp.loc[:,'Depth'].values,
                                            _wp.loc[:,'Dev'].values[:-1]
                                                    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:,'Depth'].values, tvdss=_tvd, kb=kb)


def _import_boreas1_time_depth_table(file_path: str) -> grid.TimeDepthTable:
    file_path = Path(file_path)
    assert file_path.name == 'Boreas1_vel.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0,1],
                  names=('md', 'tvdss', 'owt', 'md2', 'tvdss2', 'owt2'),
                 usecols=range(1,7))

    _twt = _td.loc[:,'owt'].values * 2 # owt to twt
    _tvdss = _td.loc[:,'tvdss'].values

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvdss)


def _import_boreas1_seismic(file_path: str) -> grid.Seismic:
    file_path = Path(file_path)
    assert file_path.name == 'Boreas1_seismic_alongwell_0_0.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
    return grid.Seismic(_seis, _twt, 'twt', name='Boreas1 seismic')




