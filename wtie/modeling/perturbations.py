"""Transformations for wavelet perturbations. """

import numpy as np

from wtie.processing import spectral
from wtie.modeling import noise
from wtie.utils.types_ import Tuple, List



class BaseTransform:
    def __init__(self, apply: bool=True, dt: float=None):
        self.apply = apply
        self.dt = dt

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Inhereting class should implement this method.
        It should take a single input (the data) and return
        a single output (the transformed data)."""
        pass

    def _verify_frequency(self, f: float):
        assert f > 0.
        assert f <= 1/(2*self.dt) # Nyquist


class Compose:
    """Apply transforms sequentially."""
    def __init__(self, transformations: List[BaseTransform],
                 random_switch: bool=False,
                 p: float=0.5):
        """
        Parameters
        ----------
        transformations : List[BaseTransform]
            List of transformations inhereting from BaseTransform.
        random_swith : bool, optional
            If set to `True` randomly switch off individual transforms.
            The default is `False`.
        p : float, optional
            probability (in [0,1]) to switch-off individual transforms.
            The default is 0.5.

        """
        self.transformations = transformations
        self.random_switch = random_switch
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        signal = np.copy(signal)

        for f in self.transformations:
            # check if there is an internal switch
            if hasattr(f, 'apply'):
                apply = f.apply

                # randomly switch off if random_swith is set to True by user
                # never apply if internal flag apply exists and is False
                if self.random_switch and apply:
                    apply = np.random.rand() > self.p

            else:
                # always apply if no internal switch
                apply = True


            if apply:
                signal = f(signal)

        return signal

    def __str__(self):
        s = ""
        for f in self.transformations:
            if "apply" in vars(f).keys():
                applied = vars(f)["apply"]
            else:
                applied = True

            if applied: s += (f.__class__.__name__ + "\n")
        return s


class RandomConstantPhaseRotation(BaseTransform):
    """Rotate input signal by constant phase."""
    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """
        Parameters
        ----------
        angle_range : tuple(float, float)
            min and max angles (in degrees) defining the random unfiform sampling
            interval.
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])


    def __call__(self, signal):
        angle = np.random.uniform(self.min_angle, self.max_angle)
        return spectral.constant_phase_rotation(signal, angle)



class RandomTimeShift(BaseTransform):
    """Time shift input signal ."""

    def __init__(self, max_samples: int, **kwargs):
        """
        Parameters
        ----------
        max_sample : int
            maximum number of samples for shifting the signal
        """
        super().__init__(**kwargs)

        self.max_samples = max_samples


    def __call__(self, signal):

        n_samples = np.random.randint(0, self.max_samples+1)

        if n_samples != 0:
            zeros = np.zeros((n_samples,))

            # left
            if np.random.randint(2) == 0:
                signal = np.concatenate((zeros, signal[:-n_samples]))
            # right
            else:
                signal = np.concatenate((signal[n_samples:], zeros))

        return signal


class TMPRandomLinearPhaseRotation(BaseTransform):
    """TODO: probably worng
    Time shift input signal by linear phase rotation."""

    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """
        Parameters
        ----------
        angle_range : tuple(float, float)
            min and max angles (in degrees) defining the random unfiform sampling
            interval.
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])


    def __call__(self, signal):

        start_angle = np.random.uniform(self.min_angle, self.max_angle)
        end_angle = np.random.uniform(start_angle, self.max_angle)

        return spectral.linear_phase_rotation(signal, start_angle, end_angle)


class RandomIndependentPhaseRotation(BaseTransform):
    """Constant phase rotation per frequency."""

    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """
        Parameters
        ----------
        angle_range : tuple(float, float)
            min and max angles (in degrees) defining the random unfiform sampling
            interval.
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])


    def __call__(self, signal):
        return spectral.random_phase_rotation(signal, self.min_angle, self.max_angle)


class RandomSimplextPhaseRotation(BaseTransform):
    """Constant phase rotation per frequency."""

    def __init__(self, max_abs_angle: int=60, scale_percentage_factor: float=2.5, **kwargs):
        """ """
        super().__init__(**kwargs)

        self.max_abs_angle = max_abs_angle
        self.scale_percentage_factor = scale_percentage_factor

    def __call__(self, signal):

        max_abs_angle_ = np.random.randint(5,self.max_abs_angle+1)

        modified = spectral.random_simplex_phase_rotation(signal,
                                                          max_abs_angle=max_abs_angle_,
                                                          scale_percentage_factor=self.scale_percentage_factor)

        pad = signal.size - modified.size
        if pad > 0:
            modified = np.concatenate((modified, np.zeros((pad,),
                                                          dtype=modified.dtype)))

        return modified






class RandomSimplexNoise(BaseTransform):
    """Add Perlin-like noise to signal."""

    def __init__(self,
                 scale_range: Tuple[float, float],
                 variation_scale_range: Tuple[float, float],
                 octaves: int=3, **kwargs):
        super().__init__(**kwargs)

        self.scale_range = scale_range
        self.variation_scale_range = variation_scale_range
        self.octaves = octaves

    def __call__(self, signal):

        #octaves = np.random.randint(self.octave_range[0], self.octave_range[1] + 1)
        scale = np.random.uniform(self.scale_range[0],
                                      self.scale_range[1])

        var_scale = np.random.uniform(self.variation_scale_range[0],
                                      self.variation_scale_range[1])

        noise_ = noise.open_simplex_noise(signal.size,
                                          scale,
                                          octaves=self.octaves,
                                          variation_scale=var_scale)

        if np.random.randint(2) == 1:
            noise_ = noise_[::-1]

        if np.random.randint(2) == 1:
            noise_ *= -1.0

        noise_ -= noise_.mean()

        return (signal + noise_)


class RandomWhitexNoise(BaseTransform):
    """Add Perlin-like noise to signal."""

    def __init__(self, scale: float, **kwargs):
        super().__init__(**kwargs)

        self.scale = scale

    def __call__(self, signal):

        noise_ = noise.white_noise(signal.size, self.scale)

        return (signal + noise_)


class RandomNotchFilter(BaseTransform):
    """Random bandcut filtering. """
    def __init__(self,
                 dt:float,
                 freq_range: Tuple[float, float],
                 band_range: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)

        self.dt = dt
        self.freq_range = freq_range
        self.band_range = band_range


    def __call__(self, signal):
        freq = np.random.uniform(self.freq_range[0], self.freq_range[1])
        band = np.random.randint(self.band_range[0], self.band_range[1]+1)
        return spectral.apply_notch_filter(signal, self.dt, freq, band, order=8)




class RandomAmplitudeScaling(BaseTransform):
    """Rotate input signal by constant phase."""
    def __init__(self, amplitude_range: Tuple[float, float], **kwargs):
        """
        Parameters
        ----------
        angle_range : tuple(float, float)
            min and max angles (in degrees) defining the random unfiform sampling
            interval.
        """
        super().__init__(**kwargs)

        self.amplitude_range = amplitude_range


    def __call__(self, signal):

        a = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        signal *= a

        return signal