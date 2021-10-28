"""FFT stuff."""

import cmath
import numpy as np
import scipy.fftpack

from scipy.signal import butter, lfilter, iirfilter

import torch
import torch.nn.functional as F


from wtie.utils.types_ import List, Tensor
from wtie.modeling.noise import open_simplex_noise


def compute_spectrum(signal: np.ndarray, dt: float,
                     to_degree: bool=True) -> List[np.ndarray]:
    """
    Parameters
    ----------
    signal : np.ndarray
        1D temporal series
    dt : float
        sampling rate in seconds [ms]

    Returns
    ----------
    freq : np.ndarray
        valid frequency range [Hz]
    ampl : np.ndarray
        amplitude spectrum
    power : np.ndarray
        power spectrum
    phase : np.ndarray
        phase spectrum [rad]
    """
    n = len(signal)
    fN = 1./(2.*dt) #Nyquist
    yf = scipy.fftpack.fft(signal)/n
    yf = yf[:n//2]
    freq = np.linspace(0.,fN,n//2)
    #freq = np.fft.rfftfreq(n, dt)

    ampl = np.abs(yf)
    power = 20*np.log10(ampl/ampl.max())

    phase = np.arctan(yf.imag / yf.real) #radian


    if to_degree:
        phase = np.rad2deg(phase)

    return freq, ampl, power, phase


def zero_phasing(signal: np.ndarray, dt: float=None) -> np.ndarray:
    """
    Parameters
    ----------
    signal : np.ndarray
        1D temporal series
    dt : float
        sampling rate in seconds [ms]

    Returns
    ----------
    zero-phased signal
    """
    n = len(signal)
    #fN = 1./(2.*dt) #Nyquist
    yf = scipy.fftpack.fft(signal)/n


    mag =  np.sqrt(yf.real**2 + yf.imag**2)
    #phase = np.arctan(yf.imag / yf.real) # <- 0

    re = mag #mag * np.cos(phase)
    #im = 0 #mag * np.sin(phase)
    yf_0 = re #+ 1.0j*im

    signal_0 = scipy.fftpack.ifft(n*yf_0)
    signal_0 = scipy.fftpack.fftshift(signal_0)

    # complex to real
    signal_0 = signal_0.astype(signal.dtype)

    return signal_0


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int):
    """
    Parameters
    ----------
    lowcut : float
        low-cut frequency of the filter [Hz]
    highcut : float
        high-cut frequency [Hz]
    fs : float
        frequency sampling [Hz]
    order : int
        order of the filter


    Returns
    ----------
    b, a : int, int
        filter parameters
    """
    nyq = 0.5 * fs
    n_low = lowcut / nyq
    n_high = highcut / nyq

    if n_low > 0 and n_high > 0:
        b, a = butter(order, [n_low, n_high], btype='band')
    elif n_low == 0 and n_high > 0:
        b, a = butter(order, n_high, btype='low', analog=False)
    elif n_low > 0 and n_high == 0:
        b, a = butter(order, n_low, btype='high', analog=False)
    else:
        raise ValueError('Frequencies must be positive and at least one non-null')

    return b, a


def apply_butter_bandpass_filter(data: np.ndarray,
                                 lowcut: float,
                                 highcut: float,
                                 fs: float,
                                 order: int=5,
                                 zero_phase: bool=True
                                 ) -> np.ndarray:
    """Filters input data with Butterworth filter."""
    if zero_phase:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order//2)
        data_tmp = lfilter(b, a, data[::-1])
        y = lfilter(b, a, data_tmp[::-1])
    else:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
    return y


def apply_butter_lowpass_filter(data: np.ndarray,
                                highcut: float,
                                fs: float,
                                order: int=5,
                                zero_phase: bool=True
                                ) -> np.ndarray:
    """Filters input data with Butterworth filter."""
    if zero_phase:
        b, a = butter_bandpass(0, highcut, fs, order=order//2)
        data_tmp = lfilter(b, a, data[::-1])
        y = lfilter(b, a, data_tmp[::-1])
    else:
        b, a = butter_bandpass(0, highcut, fs, order=order)
        y = lfilter(b, a, data)
    return y




def apply_notch_filter(data: np.ndarray, dt: float, freq: float, band: float,
                       order: int=7) -> np.ndarray:
    """Apply notch filter at frequency `freq` with bandwidth `band` """
    fs   = 1/dt
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order//2, [low, high],btype='bandstop', analog=False)

    y_tmp = lfilter(b,a, data[::-1])
    y = lfilter(b,a,y_tmp[::-1])
    return y


def phase_rotation_test(wavelet: np.ndarray, delta_theta: int = 1):
    """Determines phase of wavelet using rotation test:
    https://www.subsurfwiki.org/wiki/Phase_determination.
    Assumes wavelet is zero-centered (non-causal) and has
    constant phase at all frequencies.
    Parameters
    ----------
    wavelet : np.ndarray
        1D temporal signal
    delta_theta : int
        phase precision in degrees

    Returns
    ----------
        phase: np.ndarray
            in degree [°]
    """
    _max = 0.0
    phase = 0
    angle_range = np.arange(0, 360, step=delta_theta)
    for angle in angle_range:
        angle_rad = np.deg2rad(angle)
        w_rot = constant_phase_rotation(np.copy(wavelet), angle_rad)
        if w_rot.max() > _max:
            _max = w_rot.max()
            phase = angle

    if phase > 180:
        phase -= 360

    phase *= -1 #?

    return phase

def phase_rotation_test_at_zero_time(wavelet: np.ndarray, delta_theta: int = 1):
    """Determines phase of wavelet using [rotation test](https://www.subsurfwiki.org/wiki/Phase_determination).

    Parameters
    ----------
    wavelet : np.ndarray
        1D temporal signal
    delta_theta : int
        phase precision in degrees

    Returns
    ----------
        phase: np.ndarray
            in degree [°]
    """
    n = wavelet.size
    _max = 0.0
    phase = 0
    angle_range = np.arange(0, 360, step=delta_theta)
    for angle in angle_range:
        angle_rad = np.deg2rad(angle)
        w_rot = constant_phase_rotation(np.copy(wavelet), angle_rad)
        if w_rot[n//2].max() > _max:
            _max = w_rot[n//2].max() #TO CHECK odd vs even
            phase = angle

    if phase > 180:
        phase -= 360

    phase *= -1 #?

    return phase


def varying_phase_rotation_test_at_zero_time(wavelet,dt, delta_f=11, order=5):
    """CAREFUL: UGLY..."""
    n = wavelet.size
    fs = (1/dt)
    fN = fs/2
    freq = np.linspace(0.,fN,n//2)

    phase = np.zeros((freq.size,))

    for i,f in enumerate(freq):
        bp =  apply_butter_bandpass_filter(wavelet,
                                           lowcut=max(1,f-delta_f),
                                           highcut=min(f+delta_f, fN-1),
                                           fs=fs,
                                           order=order,
                                           zero_phase=True)

        phase_f = phase_rotation_test_at_zero_time(bp)
        phase[i] = phase_f # in degrees

    return freq, phase




def constant_phase_rotation(signal: np.ndarray, angle: float) -> np.ndarray:
    """
    Parameters
    ----------
    signal : np.ndarray
        1D temporal signal
    angle : float
        phase rotation angle [radian]
    """
    signalFFT = np.fft.rfft(signal)
    # Phase Shift the signal by angle in radian
    newSignalFFT = signalFFT * cmath.rect( 1., angle)
    newSignal = np.fft.irfft(newSignalFFT)
    return newSignal


def linear_phase_rotation(signal: np.ndarray,
                          start_angle: float,
                          end_angle: float
                          ) -> np.ndarray:
    """
    Linear phase rotation (time shift).

    Parameters
    ----------
    signal : np.ndarray
        1D temporal signal
    start_angle : float
        phase rotation angle for first frequency [radian]
    end_angle : float
        phase rotation of last frequency [radian]
    """
    signalFFT = np.fft.rfft(signal)
    newSignalFFT = np.zeros_like(signalFFT)
    rads = np.linspace(start_angle,end_angle,num=len(signalFFT))
    for i in range(len(signalFFT)):
        newSignalFFT[i] = signalFFT[i] * cmath.rect(1., rads[i])
    newSignal = np.fft.irfft(newSignalFFT)
    return newSignal







def random_phase_rotation(signal: np.ndarray,
                          min_angle: float,
                          max_angle: float
                          ) -> np.ndarray:
    """
    Random phase rotation per frequency.

    Parameters
    ----------
    signal : np.ndarray
        1D temporal signal
    min_angle : float
        minimum phase rotation angle [radian]
    max_angle : float
        maximum phase rotation [radian]
    """
    signalFFT = np.fft.rfft(signal)
    newSignalFFT = np.zeros_like(signalFFT)
    for i in range(len(signalFFT)):
        rad = np.random.uniform(min_angle,max_angle)
        newSignalFFT[i] = signalFFT[i] * cmath.rect(1., rad)
    newSignal = np.fft.irfft(newSignalFFT)
    return newSignal


def random_simplex_phase_rotation(signal: np.ndarray,
                                  max_abs_angle: int=60,
                                  scale_percentage_factor: float=2.5
                                  ) -> np.ndarray:
    """
    Random simplex phase rotation per frequency.

    Parameters
    ----------
    signal : np.ndarray
        1D temporal signal
    max_abs_angle : int, optional
        maximum absolute perturbation angle [degrees]
    scale_percentage_factor : int, optional
        controls the variation scale of the noise, smaller number means smoother
        variations. Percentage.
    """
    signalFFT = np.fft.rfft(signal)
    newSignalFFT = np.zeros_like(signalFFT)

    delta_phi = open_simplex_noise(signalFFT.size,
                                   max_abs_angle,
                                   octaves=2,
                                   variation_scale=signalFFT.size*(scale_percentage_factor/100))


    for i in range(len(signalFFT)):
        rad = np.deg2rad(delta_phi[i])
        newSignalFFT[i] = signalFFT[i] * cmath.rect(1., rad)


    newSignal = np.fft.irfft(newSignalFFT)

    return newSignal



def pt_convolution(wavelets: Tensor,
                   reflectivities: Tensor,
                   ) -> Tensor:
    """Convolution (in the signal processing sense) between a batch
    of wavelets and a batch of reflectivities.

    Parameters
    ----------
    wavelets : Tensor
        shape is (batch_size, 1, length_wavelet)
    reflectivities : Tensor
        shape is (batch_size, 1, length_reflectivity)
    mode : str, optional
        boundary conditions can be set to `valid` or `same` (see scipy).
        Default is `valid`. DISABLED.

    Returns
    ----------
    seismic: Tensor
        shape is (batch_size, 1, length_seismic), where *length_seismic* depends
        on the boundary condition `mode`.
        For `same`, *length_seismic = length_reflectivity > length_wavelet*.

    .. Warning::
        Since the function does not use batch paralellism, it is **VERY SLOW**...
    """
    wavelets = torch.unsqueeze(wavelets, 1) #[batch, 1, 1, length_wavelet]
    seismics = []
    for idx in range(wavelets.shape[0]):
        r = reflectivities[idx:idx+1] #[1, 1, length_wavelet]
        w = wavelets[idx] #[1, 1, length_wavelet]

        #r_pad = F.pad(r, (pad,pad))
        # don't forget to flip w for conv (and not corr)
        s = F.conv1d(r, w.flip(-1), stride=1, padding=0) #valid conv
        seismics.append(s)

    seismics = torch.stack(seismics) #[batch, 1, 1, length_seismic]
    seismics = seismics.squeeze(1) #[batch, 1, length_seismic]

    return seismics







