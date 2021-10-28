"""Utils functions to compute relfectivity from logs."""

import numpy as np
from numba import njit, prange



def vertical_acoustic_reflectivity(vp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Acoustic reflectivity, given Vp and rho logs.
    Parameters:
    ----------
        vp : P-wave velocity log.
        rho : Bulk density log.
    Returns:
    --------
        R0 coefficient series for vertical incidence.
    """
    upper = vp[:-1] * rho[:-1]
    lower = vp[1:] * rho[1:]
    return (lower - upper) / (lower + upper)


@njit()
def zoeppritz_rpp(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, theta: float
                  ) -> np.ndarray:
    """
    Zoeppritz Rpp reflection coefficiencients at given angle.

    Sources:
    - Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    - https://pylops.readthedocs.io/en/latest/_modules/pylops/avo/avo.html#zoeppritz_pp
    - github.com/agile-geoscience/bruges/blob/master/bruges/reflection/reflection.py

    Args:
        vp : P-wave velocity log.
        vs : S-wave velocity log.
        rho : density log.
        theta : Incidence angle in degrees.
    Returns:
        Zoeppritz solution for P-P reflectivity at incidence angle.
    """
    theta = np.deg2rad(theta)
    theta1 = theta

    # upper and lower
    vp1 = vp[:-1]; vp2 = vp[1:]
    vs1 = vs[:-1]; vs2 = vs[1:]
    rho1 = rho[:-1]; rho2 = rho[1:]

    p = np.sin(theta) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1.0 - 2.0 * np.sin(phi2)**2) - rho1 * (1.0 - 2.0 * np.sin(phi1)**2)
    b = rho2 * (1.0 - 2.0 * np.sin(phi2)**2) + 2.0 * rho1 * np.sin(phi1)**2
    c = rho1 * (1.0 - 2.0 * np.sin(phi1)**2) + 2.0 * rho2 * np.sin(phi2)**2
    d = 2.0 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1)/vp1 * np.cos(phi2)/vs2
    H = a - d * np.cos(theta2)/vp2 * np.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1/D) * (F*(b*(np.cos(theta1)/vp1) - c*(np.cos(theta2)/vp2)) \
                   - H*p**2 * (a + d*(np.cos(theta1)/vp1)*(np.cos(phi2)/vs2)))

    return rpp

@njit(parallel=False)
def prestack_rpp(vp: np.ndarray,
                 vs: np.ndarray,
                 rho: np.ndarray,
                 theta_start: int,
                 theta_end: int,
                 delta_theta: int=2
                 ) -> np.ndarray:
    angles = np.arange(theta_start, theta_end + delta_theta, delta_theta)

    rpp = np.empty(shape=(len(angles), vp.size-1)) # channel first
    rpp[:] = np.nan
    for i in prange(len(angles)):
        rpp[i,:] = zoeppritz_rpp(vp, vs, rho, angles[i])
    return rpp