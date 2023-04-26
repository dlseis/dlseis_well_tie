import warnings

from multimethod import multimethod
from collections import namedtuple


import numpy as np
import pandas as pd

from scipy.interpolate import interp1d as _interp1d


from wtie.utils.types_ import _sequence_t, Tuple, FunctionType, Dict, Union
from wtie.processing.spectral import apply_butter_lowpass_filter
from wtie.processing.sampling import downsample as _downsample


##############################
# CONSTANTS
##############################
TWT_NAME: str = 'TWT [s]'
MD_NAME: str = 'MD (kb) [m]'
TVDSS_NAME: str = 'TVDSS (MSL) [m]'
TVDKB_NAME: str = 'TVDKB [m]'
TIMELAG_NAME: str = 'Lag [s]'
ZLAG_NAME: str = 'Lag [m]'
ANGLE_NAME: str = 'Angle [°]'

_NAMES_DICT = {'twt': TWT_NAME,
               'md': MD_NAME,
               'tvdss': TVDSS_NAME,
               'tvdkb': TVDKB_NAME,
               'tlag': TIMELAG_NAME,
               'zlag': ZLAG_NAME,
               'angle': ANGLE_NAME
               }

EXISTING_BASIS_TYPES = _NAMES_DICT


def _inverted_name(name): return list(_NAMES_DICT.keys())[
    list(_NAMES_DICT.values()).index(name)]


##############################
# CLASSES
##############################
class BaseObject:
    def __init__(self, basis_type):
        self.basis_type = _NAMES_DICT[basis_type]

        # basis boolean
        self.is_twt = self.basis_type == TWT_NAME
        self.is_md = self.basis_type == MD_NAME
        self.is_tvdss = self.basis_type == TVDSS_NAME
        self.is_tvdmsl = self.basis_type == TVDSS_NAME
        self.is_tvdkb = self.basis_type == TVDKB_NAME
        self.is_tlag = self.basis_type == TIMELAG_NAME
        self.is_zlag = self.basis_type == ZLAG_NAME


class BaseTrace(BaseObject):
    """1D trace together with the sampling coordinates.
    Unit of basis is expected to be the default SI (i.e. meters or seconds).

    Do NOT instanciate directly this object. Use instead the children classes.
    """

    def __init__(self,
                 values: np.ndarray,
                 basis: np.ndarray,
                 basis_type: str,
                 name: str = None,
                 unit: str = None,
                 allow_nan: bool = True
                 ):
        """
        Parameters
        ----------
        values : np.ndarray
            Amplitude values of the trace.
        basis : np.ndarray
            Amplitude values of the basis.
        basis_type : str
            Type of the basis (e.g. two-way-time). Allowed typed are listed in the variable
            `grid.EXISTING_BASIS_TYPES`. Units must be the same as specified.
        name : str, optional
            Name of the object.
        unit : str, optional
            Unit of the amplitude values (e.g m/s for a Vp log).
        allow_nan : bool, optional
            If set to false, will raise a ValueError in case of missing values
            in the trace.

        Example
        -------
        seismic_values = np.random.normal(size=(101,))
        seismic_basis = 1.2 + np.arange(101)*0.004 # in seconds
        basis_type = 'twt' # two-way-time

        my_obj = BaseTrace(seismic_values, seismic_basis, basis_type, name='my obj')

        """
        super().__init__(basis_type)

        self._name = name
        self.unit = unit
        self.allow_nan = allow_nan

        self.is_prestack = False

        # verify shape
        assert values.ndim == basis.ndim == 1
        assert values.size == basis.size

        self.series = pd.Series(data=values,
                                name=name,
                                index=pd.Index(data=basis,
                                               name=self.basis_type)
                                )
        # verify nans
        if not allow_nan:
            assert not np.isnan(values).any()

        # verify constant sampling
        sampling = self.basis[1:] - self.basis[:-1]
        assert np.allclose(sampling, sampling[0], atol=1e-3)

        # geom attributes
        self.sampling_rate = self.basis[1] - self.basis[0]
        self.size = self.basis.size
        self.shape = self.values.shape
        self.duration = self.basis[-1] - self.basis[0]

    def time_slice(self, tmin: float, tmax: float) -> "BaseTrace":
        assert tmin >= self.basis[0] - self.sampling_rate / 2
        assert tmax <= self.basis[-1] + self.sampling_rate / 2
        idx_min = np.argmin(np.abs(self.basis - tmin))
        idx_max = np.argmin(np.abs(self.basis - tmax)) + 1
        new_basis = self.basis[idx_min:idx_max]
        new_values = self.values[idx_min:idx_max]
        return type(self)(new_values,
                          new_basis,
                          _inverted_name(self.basis_type),
                          name=self.name,
                          unit=self.unit,
                          allow_nan=self.allow_nan)

    @property
    def values(self) -> np.ndarray:
        return self.series.values

    @property
    def basis(self) -> np.ndarray:
        return self.series.index.values
    
    @basis.setter
    def basis(self, new_value):
        self.series.set_axis(new_value,inplace=True)

    def __len__(self):
        return self.size

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
        self.series.name = new_name

    def __str__(self):
        return str(self.series)

    @multimethod
    def __add__(self, other: "BaseTrace"):
        assert self.basis_type == other.basis_type
        assert np.allclose(self.basis, other.basis)
        new_values = self.values + other.values
        return type(self)(new_values, self.basis,
                          _inverted_name(self.basis_type))

    @multimethod
    def __add__(self, other: np.ndarray):
        new_values = self.values + other
        return type(self)(new_values, self.basis,
                          _inverted_name(self.basis_type))

    def __mul__(self, scalar: float):
        new_values = scalar * self.values
        return type(self)(new_values, self.basis,
                          _inverted_name(self.basis_type),
                          name=self.name)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    # @multimethod
    # def __radd__(self, other):
        # I can't make it work...


class Log(BaseTrace):
    def __init__(self, values, basis, basis_type, **kwargs):
        super().__init__(values, basis, basis_type, **kwargs)


class Reflectivity(BaseTrace):
    def __init__(self, values, basis, basis_type=None, theta: int = 0, **kwargs):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, 'twt', **kwargs)

        # incidence angle in degrees
        self.theta = theta


class Seismic(BaseTrace):
    def __init__(self, values, basis, basis_type, theta: int = 0, **kwargs):
        super().__init__(values, basis, basis_type, **kwargs)

        # incidence angle in degrees
        self.theta = theta

        # for compat with PreStackSeismic
        self.angle_range = None


WaveletUncertainties = namedtuple('WaveletUncertainties',
                                  ('ff', 'ampl_mean', 'ampl_std',
                                   'phase_mean', 'phase_std'))


class Wavelet(BaseTrace):
    def __init__(self, values, basis, basis_type=None, theta: int = 0,
                 uncertainties: WaveletUncertainties = None, **kwargs):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, 'twt', **kwargs)

        # incidence angle in degrees
        self.theta = theta

        self.uncertainties_ = uncertainties

    @property
    def uncertainties(self):
        return self.uncertainties_

    @uncertainties.setter
    def uncertainties(self, values: WaveletUncertainties):
        self.uncertainties_ = values


class DynamicLag(BaseTrace):
    def __init__(self, values, basis, basis_type, **kwargs):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, basis_type, **kwargs)


class XCorr(BaseTrace):
    def __init__(self, values, basis, basis_type, **kwargs):
        assert (basis_type == _inverted_name(TIMELAG_NAME)) or \
            (basis_type == _inverted_name(ZLAG_NAME))
        super().__init__(values, basis, basis_type, **kwargs)

    @property
    def lag(self) -> float:
        lag_idx = np.argmax(self.values)
        return self.basis[lag_idx]

    @property
    def R(self) -> float:
        #_max = self.values.max()
        #_min = self.values.min()
        # return _max if (abs(_max) >= abs(_min)) else _min
        return self.values.max()

    @property
    def Rc(self) -> float:
        return self.values[self.size//2]


class DynamicXCorr(BaseObject):
    def __init__(self, dxcorr, basis, basis_type, name=None):
        super().__init__(basis_type)
        self.name = name

        self.values = dxcorr
        self.basis = basis

        self.sampling_rate = basis[1] - basis[0]
        self.shape = dxcorr.shape

        mid_ = dxcorr.shape[1] // 2
        self.lags_basis = self.sampling_rate * \
            np.arange(-mid_, mid_) + self.sampling_rate

        if self.is_twt:
            self.lag_type = TIMELAG_NAME
        elif self.is_tvdss or self.is_md:
            self.lag_type = ZLAG_NAME
        else:
            raise ValueError


class BasePrestackTrace:
    """Collection of `Trace` objects at different angles. For simplicity
    only single azimuth angle gathers are considered."""

    def __init__(self, traces: Tuple[BaseTrace], name: str = None):
        """
        Parameters
        ----------
        traces : Tuple[BaseTrace]
            Tuple of `BaseTrace` objects. Each trace must have a unique angle
            stored in the variable trace.theta.
        """
        self.name = name

        self.is_prestack = True

        # Short cuts
        self.basis_type = traces[0].basis_type
        self._basis = traces[0].basis
        self.sampling_rate = traces[0].sampling_rate
        self.is_md = traces[0].is_md
        self.is_twt = traces[0].is_twt
        self.is_tvdss = traces[0].is_tvdss
        self.is_tvdmsl = traces[0].is_tvdmsl
        self.is_tvdkb = traces[0].is_tvdkb

        # all same basis?
        for trace in traces:
            self._verify_trace(trace)

        # angles
        angles = np.array([trace.theta for trace in traces], dtype=int)
        self._verify_angles(angles)

        self.traces = traces
        self.angles = angles
        self.delta_theta = angles[1] - angles[0]
        self.angle_range = (angles[0], angles[-1], self.delta_theta)
        self.num_angles = angles.size

        self.shape = (len(self.traces), len(self.traces[0]))
        self.trace_shape = self.traces[0].shape
        self.trace_size = self.traces[0].size
        self.num_traces = len(traces)

    def __getitem__(self, theta: int) -> BaseTrace:
        """Select trace based on angle value in degrees.
        Use self.traces to select based on index."""
        assert theta in self.angles, f"Angle {theta}° not in gather."
        idx = np.where(self.angles == theta)[0].item()
        return self.traces[idx]

    def _verify_trace(self, trace: BaseTrace):
        assert self.basis_type == trace.basis_type
        assert np.allclose(self.basis, trace.basis)

    def _verify_angles(self, angles: np.ndarray):
        dtheta = angles[1] - angles[0]
        assert dtheta > 0
        for i in range(len(angles) - 1):
            assert angles[i+1] - \
                angles[i] == dtheta, "angle sampling must be constant."

    @property
    def values(self) -> np.ndarray:
        # (angles, samples)
        values = np.empty(self.shape, dtype=float)

        for i, ref in enumerate(self.traces):
            values[i, :] = ref.values

        return values
    
    @property
    def basis(self) -> np.ndarray:
        return self._basis
    
    @basis.setter
    def basis(self, new_basis):
        self._basis = new_basis
        for trace in self.traces:
            trace.basis = new_basis
            
    @property
    def basis(self) -> np.ndarray:
        return self._basis
    
    @basis.setter
    def basis(self, new_basis):
        self._basis = new_basis
        for trace in self.traces:
            trace.basis = new_basis            
    
            
    @staticmethod
    def decimate_angles(trace: 'BasePrestackTrace',
                        start_angle: int,
                        end_angle: int,
                        delta_angle: int
                        ) -> 'BasePrestackTrace':

        assert start_angle in trace.angles
        assert end_angle in trace.angles
        assert delta_angle >= trace.delta_theta

        new_angles = range(start_angle, end_angle+delta_angle, delta_angle)

        new_trace = [trace[theta] for theta in new_angles]
        return type(trace)(new_trace, name=trace.name)

    @staticmethod
    def partial_stacking(ps_trace: 'BasePrestackTrace',
                         n: int
                         ) -> 'BasePrestackTrace':
        """Creates new traces by stacking with the `n` neighbours to the left
        and `n` to the right."""
        assert n >= 1
        assert n < ps_trace.angles.size
        num_angles = ps_trace.shape[0]
        new_values = np.zeros_like(ps_trace.values)

        # new values
        for i in range(num_angles):
            count = 0
            new_value = np.zeros_like(ps_trace.traces[0].values)
            for j in range(max(0, i-n), min(num_angles-1, i+n+1)):
                new_value += ps_trace.values[j, :]
                count += 1
            new_values[i, :] = new_value / count

        # traces objects
        new_traces = []
        trace_type = type(ps_trace.traces[0])
        trace_basis = ps_trace.traces[0].basis
        trace_basis_type = ps_trace.traces[0].basis_type
        for i, theta in enumerate(ps_trace.angles):
            new_trace = trace_type(new_values[i, :], trace_basis,
                                   _inverted_name(trace_basis_type), theta=theta,
                                   name=ps_trace.traces[i].name)

            new_traces.append(new_trace)

        return type(ps_trace)(new_traces, name=ps_trace.name)

    def __str__(self):
        txt = ("Prestack %s traces from %d to %d degrees" %
               (self.name, self.angles[0], self.angles[-1]))
        return txt


class PreStackReflectivity(BasePrestackTrace):
    def __init__(self,
                 reflectivities: Tuple[Reflectivity],
                 name: str = 'P-P reflectivity'):
        super().__init__(reflectivities, name=name)


class PreStackSeismic(BasePrestackTrace):
    def __init__(self,
                 seismics: Tuple[Seismic],
                 name: str = 'Angle gather'):
        super().__init__(seismics, name=name)


class PreStackWavelet(BasePrestackTrace):
    def __init__(self,
                 wavelets: Tuple[Wavelet],
                 name: str = 'Prestack wavelet'):
        super().__init__(wavelets, name=name)


class PreStackXCorr(BasePrestackTrace):
    def __init__(self, traces: Tuple[XCorr],
                 name: str = 'Prestack normalized x-correlation'):
        super().__init__(traces, name=name)
        self.is_tlag = traces[0].is_tlag
        self.is_zlag = traces[0].is_zlag

    @property
    def lag(self) -> np.ndarray:
        lag_indices = np.argmax(self.values, axis=-1)
        return np.array([self.basis[idx] for idx in lag_indices])

    @property
    def R(self) -> np.ndarray:
        return np.abs(self.values).max(axis=-1)

    @property
    def Rc(self) -> np.ndarray:
        return self.values[:, self.trace_size//2]


# Union types
seismic_t = Union[Seismic, PreStackSeismic]
ref_t = Union[Reflectivity, PreStackReflectivity]
wlt_t = Union[Wavelet, PreStackWavelet]
xcorr_t = Union[XCorr, PreStackXCorr]
trace_t = Union[BaseTrace, BasePrestackTrace]


# @dataclass
class LogSet:
    """Class to store together `Log` objects that belong to the same well
    and have the same basis. For simplicity enforce the use of the 3 follwoing
    keys: 'Vp', 'Rho' and 'Vs'. Logs stored under those keys will be the ones
    used in the well tie process.
    """
    # The follwoing key convention must be followed
    mandatory_keys = ['Vp', 'Rho']
    optional_keys = ['Vs', 'GR', 'Cali']  # , no in use so far

    def __init__(self, logs: Dict[str, Log]):
        """
        Parameters
        ----------
        logs : Dict[str, Log]
            Dictionary, the key represents the log type and the item the
            corresponding `Log` object. The disct must at leat contain the
            keys listed in the member vairable `mandatory_keys`. In additon,
            the key 'Vs' is required for prestack well-tie.
        """
        # logs dict must at least contain the keys 'Vp' and 'Rho'
        for key in LogSet.mandatory_keys:
            assert key in logs.keys()

        # Short cut
        self.basis_type = logs['Vp'].basis_type
        self.basis = logs['Vp'].basis
        self.sampling_rate = logs['Vp'].sampling_rate
        self.is_md = logs['Vp'].is_md
        self.is_twt = logs['Vp'].is_twt
        self.is_tvdss = logs['Vp'].is_tvdss
        self.is_tvdmsl = logs['Vp'].is_tvdmsl
        self.is_tvdkb = logs['Vp'].is_tvdkb

        # Verify basis
        for log in logs.values():
            self._verify_log(log)

        self.Logs = logs

        # Dataframe
        self.df = None
        self._create_or_update_df()

        # Short cuts
        # captial letters for Log object
        self.Vp = logs['Vp']
        self.Rho = logs['Rho']
        self.Vs = logs['Vs'] if 'Vs' in logs.keys() else None

        # small letter for numpy values
        self.vp = self.Vp.values
        self.rho = self.Rho.values
        self.vs = None if self.Vs is None else self.Vs.values

    def _verify_log(self, log: Log):
        assert self.basis_type == log.basis_type
        assert np.allclose(self.basis, log.basis)

    def _create_or_update_df(self):
        _log_dict = {}
        for name, log in self.Logs.items():
            _log_dict[name] = log.values

        df = pd.DataFrame.from_dict(
            dict(_log_dict, **{self.basis_type: self.basis}))
        df.set_index(self.basis_type, inplace=True)
        self.df = df

    def __getitem__(self, key: str):
        return self.Logs[key]

    def __setitem__(self, log: Log, key: str):
        self._verify_log(log)
        assert key not in self.Logs.keys()
        self.Logs[key] = log
        self._create_or_update_df()
        if key == 'Vs':
            self.Vs = log
            self.vs = log.values

    @property
    def AI(self) -> Log:
        """Acoustic impedence"""
        ai = self.vp * self.rho
        return Log(ai, self.basis, _inverted_name(self.basis_type), name='AI')

    @property
    def ai(self) -> np.ndarray:
        """Acoustic impedence"""
        return self.AI.values

    @property
    def Vp_Vs_ratio(self):
        """P ratio"""
        assert self.vs is not None, "You did not provide a Vs log."
        assert (self.vs > 1e-8).all(), "There are null/negative values in the Vs log."
        ratio = self.vp / self.vs
        return Log(ratio, self.basis, _inverted_name(self.basis_type), name='Vp/Vs')

    @property
    def vp_vs_ratio(self):
        return self.Vp_Vs_ratio.values

    def __str__(self):
        s_log = ("%d logs" % len(self.Logs))
        s_basis = (" in %s" % self.basis_type)
        s_shape = (" of length %d." % self.Vp.size)

        return s_log + s_basis + s_shape


class TimeDepthTable:
    """Time Depth Table. TVDSS [m] vs TWT [s]."""

    def __init__(self, twt: _sequence_t, tvdss: _sequence_t):
        """
        Parameters
        ----------
        twt : _sequence_t
            Two-way travel-time [s]
        tvdss : _sequence_t
            True vertical depth w.r.t sea surface (mean sea level)
            (corrected for well trajectory) [m]
        """
        dtype = float

        assert len(twt) == len(tvdss)
        twt = np.array(twt, dtype=dtype)
        tvdss = np.array(tvdss, dtype=dtype)

        # verify series are always (non-strictly) increasing
        assert ((tvdss[1:] - tvdss[:-1]) >= 0).all()
        assert ((twt[1:] - twt[:-1]) > 0).all()

        self.table = _create_dataframe((twt, tvdss), (TWT_NAME, TVDSS_NAME))

    @property
    def twt(self) -> np.ndarray:
        return self.table.loc[:, TWT_NAME].values

    @property
    def tvdss(self) -> np.ndarray:
        return self.table.loc[:, TVDSS_NAME].values

    @property
    def size(self):
        return self.twt.size

    def __len__(self):
        return self.size

    def slope_velocity_twt(self, dt: float = 0.004) -> Log:
        """Before computing the velocity, the table must be interpolated to
        a regular sampling.
        """
        table = TimeDepthTable.temporal_interpolation(self, dt)
        slope = self._compute_slope_from_table(table)
        return Log(slope, table.twt[1:], 'twt', name='Slope velocity')

    def slope_velocity_tvdss(self, dz: float = 5) -> Log:
        """Before computing the velocity, the table must be interpolated to
        a regular sampling.
        """
        table = TimeDepthTable.depth_interpolation(self, dz)
        slope = self._compute_slope_from_table(table)
        return Log(slope, table.tvdss[1:], 'tvdss', name='Slope velocity')

    def _compute_slope_from_table(self, table: "TimeDepthTable") -> np.ndarray:
        tvd_seg = table.tvdss[1:] - table.tvdss[:-1]
        twt_sampling = table.twt[1:] - table.twt[:-1]
        slope = 2.0*tvd_seg / twt_sampling  # 2 accounts for two-way-time
        return slope

    @staticmethod
    def z_bulk_shift(table: "TimeDepthTable", z: float) -> "TimeDepthTable":
        return TimeDepthTable(table.twt, table.tvdss + z)

    @staticmethod
    def t_bulk_shift(table: "TimeDepthTable", t: float) -> "TimeDepthTable":
        return TimeDepthTable(table.twt + t, table.tvdss)

    def temporal_interpolation(self, dt: float, mode: str = "linear") -> "TimeDepthTable":
        """Interpolate TD curves to desired time sampling.
        Parameters
        ----------
        dt : float
            constant time sampling in seconds.
        mode : intepolation mode
            see scipy doc of scipy.interpolate.interp1d
            ('linear', 'nearest', ...)
        Returns
        ----------
        A new instance of `TimeDepthTable`.
        """
        current_twt = self.twt
        current_tvd = self.tvdss

        new_twt = np.arange(current_twt[0], current_twt[-1] + dt, dt)
        # interp = _interp1d(current_twt, current_tvd, kind=mode,
        # bounds_error=False, fill_value=current_tvd[-1])

        interp = _interp1d(current_twt, current_tvd, kind=mode,
                           bounds_error=False, fill_value="extrapolate")

        new_tvd = interp(new_twt)

        return TimeDepthTable(new_twt, new_tvd)

    def depth_interpolation(self, dz: float, mode: str = "linear") -> "TimeDepthTable":
        """Interpolate TD curves to desired depth sampling.
        Parameters
        ----------
        dz : float
            constant depth sampling in meters.
        mode : intepolation mode
            see scipy doc of scipy.interpolate.interp1d
            ('linear', 'nearest', ...)
        Returns
        ----------
        A new instance of `TimeDepthTable`.
        """
        current_twt = self.twt
        current_tvd = self.tvdss

        new_tvd = np.arange(current_tvd[0], current_tvd[-1] + dz, dz)
        # interp = _interp1d(current_tvd, current_twt, kind=mode,
        # bounds_error=False, fill_value=current_twt[-1])

        interp = _interp1d(current_tvd, current_twt, kind=mode,
                           bounds_error=False, fill_value="extrapolate")

        new_twt = interp(new_tvd)
        
        return TimeDepthTable(new_twt, new_tvd)

    def __str__(self):
        table = self.table
        return ("Time-Depth table (%s vs %s) with %d entries." %
                (table.columns[0], table.columns[1], table.shape[0]))

    @staticmethod
    def get_tvdss_twt_relation_from_vp(Vp: Log,
                                       wp: 'WellPath' = None,
                                       origin: float = None,
                                       ) -> 'TimeDepthTable':
        # Vp should be preprocessed prior to input
        # (despiking, long range smoothing...)

        if Vp.is_md or Vp.is_tvdss:
            # md to tvdss
            if Vp.is_md:
                Vp = _convert_log_from_md_to_tvdss(Vp, wp)

            # integrate
            dz = Vp.basis[1:] - Vp.basis[:-1]
            twt = 2.0 * np.cumsum(dz / Vp.values[1:])

            # shift
            if origin is not None:
                twt += origin

            tvdss = Vp.basis[1:]

        elif Vp.is_twt:
            # integrate
            dt = Vp.basis[1:] - Vp.basis[:-1]
            # 0.5 to account for two-way-time
            tvdss = 0.5 * np.cumsum(dt * Vp.values[1:])

            # shift
            if origin:
                tvdss += origin

            twt = Vp.basis[1:]

        else:
            raise NotImplementedError(
                "%s basis type not implemented." % Vp.basis_type)

        return TimeDepthTable(twt, tvdss)

    @staticmethod
    def get_twt_start_from_checkshots(Vp: Log,
                                      wp: 'WellPath',
                                      checkshots: 'TimeDepthTable',
                                      return_error: bool = True
                                      ):
        """If checkshot table is available, one can compute the
        t_start for the alternative table obtained by integrating the sonic log.
        """
        
        # md to tvdss
        assert Vp.is_md
        Vp = _convert_log_from_md_to_tvdss(Vp, wp)

        # resample checkshots at log dz
        checkshots = checkshots.depth_interpolation(Vp.sampling_rate)

        # verify sampling
        z_error = np.abs(checkshots.tvdss - Vp.basis[0]).min()
        assert z_error < Vp.sampling_rate

        # t_start
        idx = np.argmin(np.abs(checkshots.tvdss - Vp.basis[0]))
        t_start = checkshots.twt[idx]

        return (t_start, z_error) if return_error else t_start

    @staticmethod
    def get_tvdss_start_from_checkshots(Vp: Log,
                                        checkshots: 'TimeDepthTable',
                                        return_error: bool = True
                                        ):
        """If checkshot table is available, one can compute the
        tvdss_start for the alternative table obtained by integrating the sonic log.
        """
        
        # md to tvdss
        assert Vp.is_twt

        # resample checkshots at log dz
        checkshots = checkshots.temporal_interpolation(Vp.sampling_rate)

        # verify sampling
        t_error = np.abs(checkshots.twt - Vp.basis[0]).min()
        assert t_error < Vp.sampling_rate

        # t_start
        idx = np.argmin(np.abs(checkshots.twt - Vp.basis[0]))
        tvdss_start = checkshots.tvdss[idx]

        return (tvdss_start, t_error) if return_error else tvdss_start


class WellPath:
    """Well trajectory information. To link Measured Depth to True Vertical Depth."""

    def __init__(self,
                 md: _sequence_t,
                 tvdss: _sequence_t = None,
                 kb: float = None
                 ):
        """
        Parameters
        ----------
        md : np.ndarray
            Measured depth sequence in meters.
        tvdss : np.ndarray , optional
            True vertical depth sequence in meters. Measured from sea surface /
            mean sea level. If not provided, well is assumed vertical.
        kb : float, optional
            KellyBushing height in meters.
        """

        dtype = float

        md = np.array(md, dtype=dtype)

        # kelly bushing
        self.kb = kb

        if tvdss is None:
            warnings.warn("You did not provide a true vertical depth (SS) series,\
                          the well is therefore assumed to be vertical.")
            tvdss = np.copy(md)

        # md is strictly increasing
        assert np.allclose(md[0], 0.0, rtol=1e-3)
        assert ((md[1:] - md[:-1]) > 0).all()

        is_going_upward = not ((tvdss[1:] - tvdss[:-1]) >= 0).all()
        if is_going_upward:
            warnings.warn("Decreasing tvd detected,\
                          this means the well is going upward at some point.")

        self.table = _create_dataframe((md, tvdss), (MD_NAME, TVDSS_NAME))

    def __str__(self):
        return ("Well path (MD [m] vs TVDSS [m]) with %d samples." % self.size)

    @property
    def size(self):
        return self.md.size

    def __len__(self):
        return self.size

    @property
    def tvdss(self) -> np.ndarray:
        return self.table.loc[:, TVDSS_NAME].values

    @property
    def tvdkb(self) -> np.ndarray:
        return self.tvdss + self.kb

    @property
    def md(self) -> np.ndarray:
        return self.table.loc[:, MD_NAME].values

    @staticmethod
    def get_tvdkb_from_inclination(md: _sequence_t, inclination: _sequence_t
                                   ) -> np.ndarray:
        """Inclination/deviation in degrees. Measured from the vertical."""
        assert md[0] == 0.0, "Deviation survey should start at 0 meters [MD]"
        assert len(inclination) == len(md) - 1
        md = np.array(md, dtype=float)
        alpha = np.deg2rad(np.array(inclination, dtype=float))

        md_segments = md[1:] - md[:-1]
        tvd_segments = md_segments * np.cos(alpha)

        tvd = np.concatenate((np.zeros((1,), dtype=float),
                              np.cumsum(tvd_segments)))

        return tvd

    @staticmethod
    def tvdss_to_tvdkb(tvdss: np.ndarray,
                       kb: float) -> np.ndarray:
        """Shift reference datum with Kelly Bushing.
        Must be specified in meters [m]. """
        return tvdss + kb

    @staticmethod
    def tvdkb_to_tvdss(tvdkb: np.ndarray,
                       kb: float) -> np.ndarray:
        """Shift reference datum with Kelly Bushing.
        Must be specified in meters [m]. """

        return tvdkb - kb

    def md_interpolation(self, dz: float, mode: str = 'linear'):
        """Interpolate trajectory at new constant measured depth sampling."""
        # new md
        md_start = self.md[0]
        md_end = self.md[-1]

        md_linear_dz = np.arange(md_start, md_end, dz)

        # interpolate tvd
        interp = _interp1d(self.md,
                           self.tvdss,
                           bounds_error=False,
                           fill_value=self.tvdss[-1],
                           kind=mode)

        new_tvd = interp(md_linear_dz)

        return WellPath(md=md_linear_dz, tvdss=new_tvd)


##############################
# FUNCTIONS
##############################
def update_trace_values(new_values: np.ndarray,
                        original_trace: BaseTrace
                        ) -> BaseTrace:
    return type(original_trace)(new_values,
                                original_trace.basis,
                                _inverted_name(original_trace.basis_type),
                                name=original_trace.name)


def get_matching_traces(trace1: Union[BaseTrace, BasePrestackTrace],
                        trace2: Union[BaseTrace, BasePrestackTrace]
                        ) -> Union[Tuple[BaseTrace], Tuple[BasePrestackTrace]]:
    # assert same basis
    assert trace1.basis_type == trace2.basis_type
    assert np.allclose(trace1.sampling_rate, trace2.sampling_rate, rtol=1e-4)

    # basis bound
    b_start = max(trace1.basis[0], trace2.basis[0])
    b_end = min(trace1.basis[-1], trace2.basis[-1])

    # indices bound
    idx_start_t1 = np.argmin(np.abs(trace1.basis - b_start)).item()
    idx_start_t2 = np.argmin(np.abs(trace2.basis - b_start)).item()

    idx_end_t1 = np.argmin(np.abs(trace1.basis - b_end)).item() + 1
    idx_end_t2 = np.argmin(np.abs(trace2.basis - b_end)).item() + 1

    # should be okay with small sampling rate rounding errors
    assert (idx_end_t1 - idx_start_t1) == (idx_end_t2 - idx_start_t2)

    # fork depending on type
    if issubclass(type(trace1), BaseTrace) and issubclass(type(trace2), BaseTrace):

        trace1_match = type(trace1)(trace1.values[idx_start_t1:idx_end_t1],
                                    trace1.basis[idx_start_t1:idx_end_t1],
                                    _inverted_name(trace1.basis_type),
                                    name=trace1.name)

        trace2_match = type(trace2)(trace2.values[idx_start_t2:idx_end_t2],
                                    trace2.basis[idx_start_t2:idx_end_t2],
                                    _inverted_name(trace2.basis_type),
                                    name=trace2.name)

    elif issubclass(type(trace1), BasePrestackTrace) and \
            issubclass(type(trace2), BasePrestackTrace):
        assert (trace1.angles == trace2.angles).all()
        trace1_match = []
        trace2_match = []

        for theta in trace1.angles:
            trace1_match.append(type(trace1[theta])(trace1[theta].values[idx_start_t1:idx_end_t1],
                                trace1.basis[idx_start_t1:idx_end_t1],
                                _inverted_name(trace1.basis_type),
                                name=trace1[theta].name, theta=theta)
                                )

            trace2_match.append(type(trace2[theta])(trace2[theta].values[idx_start_t2:idx_end_t2],
                                trace2.basis[idx_start_t2:idx_end_t2],
                                _inverted_name(trace2.basis_type),
                                name=trace2[theta].name, theta=theta)
                                )

        trace1_match = type(trace1)(trace1_match)
        trace2_match = type(trace2)(trace2_match)

    else:
        raise NotImplementedError

    #assert trace1_match.size == trace2_match.size
    #print(trace1_match.basis)
    #print(trace2_match.basis)
    assert np.allclose(trace1_match.basis, trace2_match.basis, rtol=1e-2)
    #DOUBLECHECK IF IT MAKES SENSE IN ALL CASES
    trace2_match.basis = trace1_match.basis
    assert np.allclose(trace1_match.basis, trace2_match.basis, rtol=1e-3)
    
    return trace1_match, trace2_match


def _lowpass_filter_trace(trace: BaseTrace,
                          highcut: float,
                          order: int = 5) -> BaseTrace:

    fs = 1 / trace.sampling_rate
    fN = fs / 2.0
    assert highcut < fN

    low_signal = apply_butter_lowpass_filter(trace.values,
                                             highcut,
                                             fs,
                                             order=order,
                                             zero_phase=True
                                             )
    return type(trace)(values=low_signal, basis=trace.basis,
                       basis_type=_inverted_name(trace.basis_type),
                       name=trace.name)


def lowpass_filter_trace(trace: Union[BaseTrace, BasePrestackTrace],
                         highcut: float,
                         order: int = 5
                         ) -> Union[BaseTrace, BasePrestackTrace]:

    if issubclass(type(trace), BaseTrace):
        return _lowpass_filter_trace(trace, highcut, order=order)
    elif issubclass(type(trace), BasePrestackTrace):
        return _apply_trace_process_to_prestack_trace(_lowpass_filter_trace,
                                                      trace,
                                                      highcut,
                                                      order=order)
    else:
        raise NotImplementedError


def _apply_trace_process_to_prestack_trace(process: FunctionType,
                                           trace: BasePrestackTrace,
                                           *args, **kwargs
                                           ) -> BasePrestackTrace:
    separate_traces = []
    for theta in trace.angles:
        that_tace = process(trace[theta], *args, **kwargs)
        that_tace.theta = theta
        separate_traces.append(that_tace)

    return type(trace)(separate_traces)


def _apply_trace_process_to_logset(process: FunctionType,
                                   logset: LogSet,
                                   *args, **kwargs) -> LogSet:
    new_logs = {}
    for name, log in logset.Logs.items():
        _log = process(log, *args, **kwargs)
        new_logs[name] = _log

    return LogSet(new_logs)


def lowpass_filter_logset(logset: LogSet,
                          highcut: float,
                          order: int = 5) -> LogSet:

    return _apply_trace_process_to_logset(lowpass_filter_trace,
                                          logset, highcut, order=order)


def downsample_trace(trace: BaseTrace,
                     new_sampling: float
                     ) -> BaseTrace:
    assert new_sampling > trace.sampling_rate
    # lowpass and decimate
    div_factor = int(round(new_sampling / trace.sampling_rate))
    #signal_resamp = _decimate(trace.values, div_factor)
    # correct for DC bias
    #signal_resamp = signal_resamp - signal_resamp.mean() + trace.values.mean()
    signal_resamp = _downsample(trace.values, div_factor)

    basis_resamp = trace.basis[::div_factor]

    return type(trace)(values=signal_resamp, basis=basis_resamp,
                       basis_type=_inverted_name(trace.basis_type),
                       name=trace.name)


def downsample_logset(logset: LogSet,
                      new_sampling: float
                      ) -> LogSet:

    return _apply_trace_process_to_logset(downsample_trace, logset, new_sampling)


def resample_logset(logset: LogSet,
                    new_sampling: float
                    ) -> LogSet:

    return _apply_trace_process_to_logset(resample_trace, logset, new_sampling)


def resample_trace(trace: Union[BaseTrace, BasePrestackTrace],
                   new_sampling: float
                   ) -> Union[BaseTrace, BasePrestackTrace]:

    if issubclass(type(trace), BaseTrace):
        return _resample_trace(trace, new_sampling)
    elif issubclass(type(trace), BasePrestackTrace):
        return _apply_trace_process_to_prestack_trace(_resample_trace,
                                                      trace,
                                                      new_sampling)
    else:
        raise NotImplementedError


def _resample_trace(trace: BaseTrace,
                    new_sampling: float
                    ) -> BaseTrace:

    sr = trace.sampling_rate

    if new_sampling < sr:
        return upsample_trace(trace, new_sampling)
    elif new_sampling > sr:
        return downsample_trace(trace, new_sampling)
    else:
        return trace


def upsample_trace(trace: BaseTrace,
                   new_sampling: float
                   ) -> BaseTrace:
    """
    sinc interpolation

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """
    # Find the period
    assert new_sampling < trace.sampling_rate
    current_sr = trace.sampling_rate

    new_basis = np.arange(trace.basis[0], trace.basis[-1], step=new_sampling)
    #new_length = int(len(trace) * new_sampling / current_sr)

    sincM = np.tile(new_basis, (len(trace), 1)) - np.tile(trace.basis[:, np.newaxis],
                                                          (1, len(new_basis)))
    new_signal = np.dot(trace.values, np.sinc(sincM / current_sr))
    return type(trace)(values=new_signal, basis=new_basis,
                       basis_type=_inverted_name(trace.basis_type),
                       name=trace.name)


def _convert_log_from_md_to_tvdss(log: Log,
                                  trajectory: WellPath,
                                  dz: float = None,
                                  interpolation: str = 'linear'
                                  ) -> Log:
    """Account for the deviation of the well."""
    # input log
    assert log.is_md

    # interpolate trajectory at same log sampling
    trajectory_at_log_dz = trajectory.md_interpolation(log.sampling_rate)

    # current tvd
    idx_start = np.argmin(
        np.abs(trajectory_at_log_dz.md - log.basis[0])).item()
    if idx_start + len(log) >= len(trajectory_at_log_dz):
        warnings.warn("Truncating log as the well path information does not reach\
                      the maximum depth.")
        max_idx = idx_start + (len(trajectory_at_log_dz) - idx_start)
        log = Log(log.values[:(len(trajectory_at_log_dz) - idx_start)],
                  log.basis[:(len(trajectory_at_log_dz) - idx_start)],
                  basis_type=_inverted_name(log.basis_type), name=log.name)

    else:
        max_idx = idx_start + len(log)

    #assert idx_start + len(log) < len(trajectory_at_log_dz)
    current_tvd = trajectory_at_log_dz.tvdss[idx_start:max_idx]

    # interpolate to linear tvd
    if dz is None:
        # keep the same sampling rate
        dz = log.sampling_rate
    linear_tvd = np.arange(current_tvd[0], current_tvd[-1]+dz, dz)

    interp = _interp1d(current_tvd, log.values,
                       bounds_error=False, fill_value=log.values[-1],
                       kind=interpolation)

    values_at_tvd_dz = interp(linear_tvd)

    return Log(values_at_tvd_dz, linear_tvd, 'tvdss',
               name=log.name, allow_nan=log.allow_nan)


def convert_log_from_md_to_twt(log: Log,
                               table: TimeDepthTable,
                               trajectory: WellPath,
                               dt: float,
                               interpolation: str = 'linear'
                               ) -> Log:
    # input log
    assert log.is_md
    dz = log.sampling_rate

    # md to tvd
    log = _convert_log_from_md_to_tvdss(log, trajectory, dz=dz,
                                        interpolation=interpolation)
    assert log.is_tvdss
    start_z = log.basis[0]  # tvd
    
    # interpolate t-d table at dz
    table_at_dz = table.depth_interpolation(dz)
    #max_table_tvdss = table_at_dz.tvdss[-1]

    # find equivalent twt
    idx_start = np.argmin(np.abs(table_at_dz.tvdss - start_z)).item()

    # truncate log if longer than table relationship
    if idx_start + len(log) >= len(table_at_dz):
        warnings.warn("Truncating log as the time-depth table does not reach\
                      the maximum depth.")
        max_idx = (len(table_at_dz) - idx_start)
        log = Log(log.values[:max_idx], log.basis[:max_idx], _inverted_name(log.basis_type),
                  name=log.name)

    log_twt = table_at_dz.twt[idx_start:idx_start+len(log)]

    # interpolate to regular dt
    linear_twt = np.arange(log_twt[0], log_twt[-1]+dt, dt)
    
    interp = _interp1d(log_twt, log.values,
                       bounds_error=False, fill_value=log.values[-1],
                       kind=interpolation)
    values_at_dt = interp(linear_twt)

    return Log(values_at_dt, linear_twt, 'twt',
               name=log.name, allow_nan=log.allow_nan)


################################
# UTILS
################################
def _create_dataframe(arrays: Tuple[np.ndarray], names: Tuple[str]) -> pd.DataFrame:
    _table = np.stack(arrays, axis=-1)
    return pd.DataFrame(_table, columns=names)
