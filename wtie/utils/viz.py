"""Plotting utils """
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import ConnectionPatch
from matplotlib import gridspec

from wtie.processing import grid
from wtie.utils.types_ import Tuple
from wtie.processing.spectral import compute_spectrum


def plot_seismics(real_seismic: grid.Seismic,
                  pred_seismic: grid.Seismic,
                  reflectivity: grid.Reflectivity,
                  normalize: bool = True,
                  figsize: Tuple[int, int] = (7, 4)
                  ) -> plt.subplots:

    assert np.allclose(real_seismic.basis, pred_seismic.basis, rtol=1e-3)
    assert np.allclose(real_seismic.basis, reflectivity.basis, rtol=1e-3)
    basis = real_seismic.basis

    if normalize:
        def f(x): return x / max(x.max(), -x.min())
    else:
        def f(x): return x

    x_picks = np.where(reflectivity.values != 0)[0]
    y_picks = f(reflectivity.values[reflectivity.values != 0.])

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    axes[0].plot(basis, np.zeros_like(basis),
                 color='r', alpha=0.5, label='Reflectivity')
    axes[0].plot(basis, f(real_seismic.values), lw=1.5, label='Real seismic')
    #axes[0].plot(ref_t, noise, 'g', alpha=0.3, lw=1.)
    for i in range(len(x_picks)):
        axes[0].vlines(basis[x_picks[i]], ymin=0,
                       ymax=y_picks[i], color='r', lw=1.)

    axes[1].plot(basis, np.zeros_like(basis), color='r', alpha=0.5)
    axes[1].plot(basis, f(pred_seismic.values),
                 lw=1.5, label='Synthetic seismic')
    for i in range(len(x_picks)):
        axes[1].vlines(basis[x_picks[i]], ymin=0,
                       ymax=y_picks[i], color='r', lw=1.)

    for ax in axes:
        ax.set_xlim([basis[0], basis[-1]])
        if normalize:
            ax.set_ylim([-1.05, 1.05])
        ax.legend(loc='best')

    axes[0].set_xlabel(reflectivity.basis_type)

    plt.tight_layout()

    return fig, axes


def plot_seismic_and_reflectivity(seismic: grid.Seismic,
                                  reflectivity: grid.Reflectivity,
                                  normalize: bool = False,
                                  figsize: Tuple[int, int] = (3, 5),
                                  fig_axes: plt.subplots = None,
                                  title: str = None
                                  ) -> plt.subplots:

    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)

    if normalize:
        seis_ = np.copy(seismic.values)
        seis_ /= np.abs(seis_).max()
        seismic = grid.update_trace_values(seis_, seismic)

        ref_ = np.copy(reflectivity.values)
        ref_ /= np.abs(ref_).max()
        reflectivity = grid.update_trace_values(ref_, reflectivity)

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    plot_reflectivity(reflectivity, fig_axes=(fig, ax))
    plot_trace(seismic, fig_axes=(fig, ax))

    if title is not None:
        ax.set_xlabel(title)
    else:
        ax.set_xlabel("")

    plt.tight_layout()

    return fig, ax


def plot_tie_window(logset: grid.LogSet,
                    reflectivity: grid.Reflectivity,
                    synthetic_seismic: grid.Seismic,
                    real_seismic: grid.Seismic,
                    xcorr: grid.XCorr,
                    dxcorr: grid.DynamicXCorr,
                    figsize: Tuple[int, int] = (7, 4),
                    wiggle_scale_syn: float = 1.0,
                    wiggle_scale_real: float = 1.0
                    ) -> plt.subplots:

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 8)

    axes = [fig.add_subplot(gs[0]),
            fig.add_subplot(gs[1]),
            fig.add_subplot(gs[2:4]),
            fig.add_subplot(gs[4:6]),
            fig.add_subplot(gs[6:7]),
            fig.add_subplot(gs[7:])
            ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    # seismic
    plot_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn, repeat_n_times=5,
                      fig_axes=(fig, axes[2]))
    plot_wiggle_trace(real_seismic, scaling=wiggle_scale_real, repeat_n_times=5,
                      fig_axes=(fig, axes[3]))

    residual = grid.Seismic(real_seismic.values - synthetic_seismic.values,
                            real_seismic.basis, 'twt', name='Residual')

    plot_wiggle_trace(residual, scaling=wiggle_scale_real, repeat_n_times=1,
                      fig_axes=(fig, axes[4]))

    # for ax in axes[2:5]:
    #    _space = 0.05*np.abs(real_seismic.values.max())
    #    ax.set_xlim((real_seismic.values.min() - _space,
    #                 (real_seismic.values.max() + wiggle_scale_real + _space)))
    #    ax.set_xticks([])

    # dxcoor
    #_,_,cbar = plot_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]))
    _, _, cbar = plot_dynamic_xcorr(dxcorr, fig_axes=(fig, axes[5]))
    axes[5].set_xlabel("Correlation")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes:
        ax.locator_params(axis='y', nbins=28)
        #ax.locator_params(axis='x', nbins=8)

    # axes[4].yaxis.tick_right()

    fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" %
                 (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.tight_layout()
    return fig, axes


def TMPplot_tie_window(logset: grid.LogSet,
                       reflectivity: grid.Reflectivity,
                       synthetic_seismic: grid.Seismic,
                       real_seismic: grid.Seismic,
                       xcorr: grid.XCorr,
                       dxcorr: grid.DynamicXCorr,
                       figsize: Tuple[int, int] = (7, 4),
                       wiggle_scale_syn: float = 1.0,
                       wiggle_scale_real: float = 1.0
                       ) -> plt.subplots:

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 7)

    axes = [fig.add_subplot(gs[0]),
            fig.add_subplot(gs[1]),
            fig.add_subplot(gs[2:4]),
            fig.add_subplot(gs[4:6]),
            fig.add_subplot(gs[6:])
            ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    # seismic
    plot_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn, repeat_n_times=5,
                      fig_axes=(fig, axes[2]))
    plot_wiggle_trace(real_seismic, scaling=wiggle_scale_real, repeat_n_times=5,
                      fig_axes=(fig, axes[3]))

    # dxcoor
    #_,_,cbar = plot_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]))
    _, _, cbar = plot_dynamic_xcorr(dxcorr, fig_axes=(fig, axes[4]))
    axes[4].set_xlabel("Correlation")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes:
        ax.locator_params(axis='y', nbins=28)
        #ax.locator_params(axis='x', nbins=8)

    # axes[4].yaxis.tick_right()

    fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" %
                 (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.tight_layout()
    return fig, axes


def plot_prestack_tie_window(logset: grid.LogSet,
                             reflectivity: grid.PreStackReflectivity,
                             synthetic_seismic: grid.PreStackSeismic,
                             real_seismic: grid.PreStackSeismic,
                             xcorr: grid.PreStackXCorr,
                             figsize: Tuple[int, int] = (7, 4),
                             decimate_wiggles: int = 2,
                             wiggle_scale_syn: float = 1.0,
                             wiggle_scale_real: float = 1.0,
                             reflectivity_scale: float = 1.0
                             ) -> plt.subplots:

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 7)

    axes = [fig.add_subplot(gs[0]),
            fig.add_subplot(gs[1]),
            fig.add_subplot(gs[2:4]),
            fig.add_subplot(gs[4:6]),
            fig.add_subplot(gs[6:])
            ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_trace(logset.Vp_Vs_ratio, fig_axes=(fig, axes[1]))

    # seismic
    plot_prestack_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn,
                               decimate_every_n=decimate_wiggles,
                               fig_axes=(fig, axes[2]))
    axes[2].set_title('Synthetic gather')
    plot_prestack_wiggle_trace(real_seismic, scaling=wiggle_scale_real,
                               decimate_every_n=decimate_wiggles,
                               fig_axes=(fig, axes[3]))
    axes[3].set_title('Real gather')

    for ax in [axes[2], axes[3]]:
        plot_prestack_reflectivity(reflectivity,
                                   scaling=reflectivity_scale,
                                   decimate_every_n=decimate_wiggles,
                                   hline_params={'color': 'k', 'lw': 1.},
                                   fig_axes=(fig, ax))

    # xcoor
    _, _, cbar = plot_prestack_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]),
                                               decimate_wiggles=decimate_wiggles)

    for ax in axes[1:-1]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes[:-1]:
        ax.locator_params(axis='y', nbins=28)
        #ax.locator_params(axis='x', nbins=8)

    axes[4].yaxis.tick_right()

    # fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % \
    #             (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.suptitle("Prestack well-tie. \nMean max correlation of %.2f at a mean lag of %.3f s (Mean Rc = %.2f)" %
                 (xcorr.R.mean(), xcorr.lag.mean(), xcorr.Rc.mean()))

    fig.tight_layout()
    return fig, axes


def NONOplot_prestack_tie_window(logset: grid.LogSet,
                                 reflectivity: grid.PreStackReflectivity,
                                 synthetic_seismic: grid.PreStackSeismic,
                                 real_seismic: grid.PreStackSeismic,
                                 xcorr: grid.PreStackXCorr,
                                 figsize: Tuple[int, int] = (7, 4),
                                 decimate_wiggles: int = 2,
                                 wiggle_scale_syn: float = 1.0,
                                 wiggle_scale_real: float = 1.0,
                                 reflectivity_scale: float = 1.0
                                 ) -> plt.subplots:

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 9)

    axes = [fig.add_subplot(gs[0]),
            fig.add_subplot(gs[1]),
            fig.add_subplot(gs[2:4]),
            fig.add_subplot(gs[4:6]),
            fig.add_subplot(gs[6:8]),
            fig.add_subplot(gs[8:])
            ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_trace(logset.Vp_Vs_ratio, fig_axes=(fig, axes[1]))

    # seismic
    plot_prestack_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn,
                               decimate_every_n=decimate_wiggles,
                               fig_axes=(fig, axes[2]))
    axes[2].set_title('Synthetic gather')
    plot_prestack_wiggle_trace(real_seismic, scaling=wiggle_scale_real,
                               decimate_every_n=decimate_wiggles,
                               fig_axes=(fig, axes[3]))
    axes[3].set_title('Real gather')

    # residual
    residual = []
    for theta in real_seismic.angles:
        residual.append(grid.Seismic(real_seismic[theta].values - synthetic_seismic[theta].values,
                                     real_seismic.basis, 'twt', name='Residual', theta=theta))
    residual = grid.PreStackSeismic(residual, name='Residual')

    plot_prestack_wiggle_trace(residual, scaling=wiggle_scale_real,
                               decimate_every_n=decimate_wiggles,
                               fig_axes=(fig, axes[4]))
    axes[4].set_title('Residual')

    for ax in [axes[2], axes[3], axes[4]]:
        plot_prestack_reflectivity(reflectivity,
                                   scaling=reflectivity_scale,
                                   decimate_every_n=decimate_wiggles,
                                   hline_params={'color': 'k', 'lw': 1.},
                                   fig_axes=(fig, ax))

    # xcoor
    _, _, cbar = plot_prestack_trace_as_pixels(xcorr, fig_axes=(fig, axes[5]),
                                               decimate_wiggles=decimate_wiggles)

    for ax in axes[1:-1]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes[:-1]:
        ax.locator_params(axis='y', nbins=28)
        #ax.locator_params(axis='x', nbins=8)

    axes[4].yaxis.tick_right()

    # fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % \
    #             (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.suptitle("Prestack well-tie. \nMean max correlation of %.2f at a mean lag of %.3f s (Mean Rc = %.2f)" %
                 (xcorr.R.mean(), xcorr.lag.mean(), xcorr.Rc.mean()))

    fig.tight_layout()
    return fig, axes


def plot_wavelet(wavelet: grid.Wavelet,
                 figsize: Tuple[int, int] = None,
                 title: str = "Predicted wavelet",
                 plot_params: dict = None,
                 fmax: float = None,
                 phi_max: float = 100,
                 abs_t_max: float = None,
                 fig_axes: plt.subplots = None,
                 ) -> plt.subplots:

    if plot_params is None:
        plot_params = {}

    if fig_axes is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)

        axes = [fig.add_subplot(gs[0, :]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1])]
    else:
        fig, axes = fig_axes

    ff, ampl, _, phase = compute_spectrum(wavelet.values,
                                          wavelet.sampling_rate,
                                          to_degree=True)

    ampl /= ampl.max()

    if fmax is None:
        fmax = ff[-1]

    axes[0].plot(wavelet.basis, wavelet.values, color='k', alpha=0.5, lw=0.5)
    axes[0].fill_between(wavelet.basis,
                         wavelet.values,
                         where=(wavelet.values >= 0.0),
                         color='b', alpha=.8, interpolate=True)
    axes[0].fill_between(wavelet.basis,
                         wavelet.values,
                         where=(wavelet.values < 0.0),
                         color='r', alpha=.8, interpolate=True)

    axes[0].plot(wavelet.basis, np.zeros_like(wavelet.basis), color='k', lw=.5)
    axes[0].set_xlim((wavelet.basis[0], wavelet.basis[-1]))
    axes[0].set_ylim((2.0*wavelet.values.min(), 1.15*wavelet.values.max()))

    if abs_t_max is not None:
        axes[0].set_xlim((-abs_t_max, abs_t_max))

    # amplitude
    axes[1].plot(ff, ampl, '-')
    axes[1].set_ylim((0.0, 1.1*ampl.max()))
    axes[1].set_xlim((ff[0], fmax))

    # phase
    axes[2].plot(ff, phase, '+')
    axes[2].plot(ff, np.zeros_like(phase), color='k',
                 lw=.5, alpha=0.5, linestyle='--')
    axes[2].set_xlim((ff[0], fmax))
    axes[2].set_ylim((-abs(phi_max), abs(phi_max)))

    # uncertainities
    if wavelet.uncertainties is not None:
        assert np.allclose(ff, wavelet.uncertainties.ff)

        # amplitude
        ampl_mean = wavelet.uncertainties.ampl_mean
        ampl_std = wavelet.uncertainties.ampl_std
        max_ = ampl_mean.max()
        ampl_mean /= max_
        ampl_std /= max_
        axes[1].fill_between(ff,
                             ampl_mean-ampl_std,
                             ampl_mean+ampl_std,
                             color='gray', alpha=0.7)

        # phase
        axes[2].plot(ff, wavelet.uncertainties.phase_mean,
                     color='gray', alpha=0.9, lw=.9)
        axes[2].fill_between(ff,
                             wavelet.uncertainties.phase_mean-wavelet.uncertainties.phase_std,
                             wavelet.uncertainties.phase_mean+wavelet.uncertainties.phase_std,
                             color='gray', alpha=0.6)

    axes[0].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Phase [°]")
    axes[2].set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    return fig, axes


def plot_prestack_wavelet(wavelet: grid.PreStackWavelet,
                          figsize: Tuple[int, int] = (9, 6),
                          three_angles: Tuple[int] = None,
                          title: str = "Predicted wavelet",
                          plot_params: dict = None,
                          fmax: float = None,
                          phi_max: float = 100
                          ) -> plt.subplots:

    if plot_params is None:
        plot_params = {}

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 3)

    axes = [fig.add_subplot(gs[:2, 0]),
            fig.add_subplot(gs[:2, 1]),
            fig.add_subplot(gs[:2, 2]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
            fig.add_subplot(gs[3, 0]),
            fig.add_subplot(gs[3, 1]),
            fig.add_subplot(gs[3, 2])]

    if three_angles is None:
        three_angles = [wavelet.angles[0],
                        wavelet.angles[wavelet.angles.size//2],
                        wavelet.angles[-1]]

    # Wavelets
    for i, ax in enumerate([axes[0], axes[1], axes[2]]):
        theta = three_angles[i]
        _values = wavelet[theta].values
        ax.plot(wavelet.basis, _values, color='k', alpha=0.5, lw=0.5)
        ax.fill_between(wavelet.basis,
                        _values,
                        where=(_values >= 0.0),
                        color='b', alpha=.8, interpolate=True)
        ax.fill_between(wavelet.basis,
                        _values,
                        where=(_values < 0.0),
                        color='r', alpha=.8, interpolate=True)

        ax.plot(wavelet.basis, np.zeros_like(wavelet.basis), color='k', lw=.5)
        ax.set_xlim((wavelet.basis[0], wavelet.basis[-1]))
        #ax.set_ylim((2.0*_values.min(), 1.15*_values.max()))
        ax.set_ylim((-0.5, 1.15))
        ax.set_title(("Wavelet at %d °" % theta))

    axes[0].set_xlabel("Time [s]")

    # Spectrum
    for i, idx in enumerate(range(3, 6, 1)):
        theta = three_angles[i]
        _values = wavelet[theta].values
        ff, ampl, _, phase = compute_spectrum(_values, wavelet.sampling_rate,
                                              to_degree=True)
        ampl /= ampl.max()

        axes[idx].plot(ff, ampl, '-')
        axes[idx].set_ylim((0.0, 1.05*ampl.max()))
        axes[idx].set_xlim((ff[0], fmax))

        axes[idx+3].plot(ff, phase, '+')
        axes[idx+3].plot(ff, np.zeros_like(phase), color='k',
                         lw=.5, alpha=0.5, linestyle='--')
        axes[idx+3].set_xlim((ff[0], fmax))
        axes[idx+3].set_ylim((-phi_max, phi_max))

        # uncertainities
        if wavelet[theta].uncertainties is not None:
            assert np.allclose(ff, wavelet[theta].uncertainties.ff)

            # amplitude
            ampl_mean = wavelet[theta].uncertainties.ampl_mean
            ampl_std = wavelet[theta].uncertainties.ampl_std
            max_ = ampl_mean.max()
            ampl_mean /= max_
            ampl_std /= max_

            axes[idx].fill_between(ff,
                                   ampl_mean-ampl_std,
                                   ampl_mean+ampl_std,
                                   color='gray', alpha=0.7)

            # phase
            axes[idx+3].plot(ff, wavelet[theta].uncertainties.phase_mean,
                             color='gray', alpha=0.9, lw=.9)
            axes[idx+3].fill_between(ff,
                                     wavelet[theta].uncertainties.phase_mean -
                                     wavelet[theta].uncertainties.phase_std,
                                     wavelet[theta].uncertainties.phase_mean +
                                     wavelet[theta].uncertainties.phase_std,
                                     color='gray', alpha=0.6)

    axes[3].set_ylabel("Normalized amplitude")
    axes[3].set_xlabel("Frequency [Hz]")
    axes[6].set_ylabel("Phase [°]")
    axes[6].set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    return fig, axes


def plot_logsets_overlay(logset1: grid.LogSet,
                         logset2: grid.LogSet,
                         figsize: Tuple[int, int] = (7, 6),
                         title: str = "Well Logs",
                         fig_axes: tuple = None
                         ) -> plt.subplots:

    assert logset1.basis_type == logset2.basis_type
    is_vs = (logset1.vs is not None) and (logset2.vs is not None)

    if fig_axes is None:
        if is_vs:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = fig_axes

    plot_logset(logset1, fig_axes=(fig, axes))
    plot_logset(logset2, fig_axes=(fig, axes),
                plot_params=dict(linewidth=1.0))

    return fig, axes


def plot_logset(logset: grid.LogSet,
                figsize: Tuple[int, int] = (7, 6),
                title: str = "Well Logs",
                plot_params: dict = None,
                fig_axes: tuple = None
                ) -> plt.subplots:

    is_vs = logset.vs is not None

    if fig_axes is None:
        if is_vs:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = fig_axes

    if plot_params is None:
        plot_params = {}

    axes[0].plot(logset.vp / 1000, logset.basis, **plot_params)

    if is_vs:
        axes[1].plot(logset.vs / 1000, logset.basis, **plot_params)
        axes[2].plot(logset.rho, logset.basis, **plot_params)
    else:
        axes[1].plot(logset.rho, logset.basis, **plot_params)

    axes[0].set_ylabel(logset.basis_type)
    axes[0].set_xlabel("Vp [km/s]")

    if is_vs:
        axes[1].set_xlabel("Vs [km/s]")
        axes[2].set_xlabel("Rho [g/cm³]")
    else:
        axes[1].set_xlabel("Rho [g/cm³]")

    for ax in axes:
        ax.set_ylim((logset.basis[0], logset.basis[-1]))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    for ax in axes[1:]:
        ax.set_yticklabels("")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_wellpath(wellpath: grid.WellPath,
                  figsize: Tuple[int, int] = (5, 5),
                  fig_axes: tuple = None
                  ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(wellpath.md, wellpath.tvdkb, lw=1.5)
    ax.plot(wellpath.md, wellpath.md, color='k', lw=0.5)
    ax.set_xlabel(grid.MD_NAME)
    ax.set_ylabel(grid.TVDKB_NAME)
    ax.set_title("Well curvature")

    ax.set_xlim((wellpath.md[0], wellpath.md[-1]))
    ax.set_ylim((wellpath.tvdkb[0], wellpath.md[-1]))
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax


def plot_td_table(table: grid.TimeDepthTable,
                  figsize: Tuple[int, int] = (4, 4),
                  plot_params: dict = None,
                  fig_axes: tuple = None
                  ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {}

    ax.plot(table.tvdss, table.twt, **plot_params)
    ax.set_xlabel(grid.TVDSS_NAME)
    ax.set_ylabel(grid.TWT_NAME)
    ax.set_title("Time-Depth table.")

    ax.set_xlim((table.tvdss[0], table.tvdss[-1]))
    ax.set_ylim((table.twt[0], table.twt[-1]))
    # ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax


def plot_reflectivity(reflectivity: grid.Reflectivity,
                      figsize: tuple = (3, 5),
                      fig_axes: tuple = None
                      ) -> plt.subplots:

    x_picks = np.where(reflectivity.values != 0)[0]
    y_picks = reflectivity.values[reflectivity.values != 0.]

    basis = reflectivity.basis

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(np.zeros_like(basis), basis, color='k', alpha=0.5, lw=0.5)
    for i in range(len(x_picks)):
        ax.hlines(basis[x_picks[i]], xmin=0, xmax=y_picks[i], color='r', lw=1.)

    ax.set_xlabel(reflectivity.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(reflectivity.basis_type)

    ax.set_ylim((basis[0], basis[-1]))
    ax.invert_yaxis()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_prestack_reflectivity(reflectivity: grid.PreStackReflectivity,
                               scaling: float = 10.0,
                               decimate_every_n: int = 1,
                               figsize: tuple = (7, 6),
                               fig_axes: tuple = None,
                               hline_params: dict = None
                               ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if hline_params is None:
        hline_params = dict(color='r', lw=1.)

    basis = reflectivity.basis

    for j, angle in enumerate(reflectivity.angles[::decimate_every_n]):
        values = scaling*reflectivity[angle].values
        x_picks = np.where(values != 0)[0]
        y_picks = values[values != 0.]

        ax.plot(np.zeros_like(basis)+angle, basis,
                color='k', alpha=0.5, lw=0.5)
        for i in range(len(x_picks)):
            ax.hlines(basis[x_picks[i]], xmin=angle,
                      xmax=y_picks[i]+angle, **hline_params)

    fig.suptitle(reflectivity.name)
    ax.set_ylabel(reflectivity.basis_type)

    ax.set_ylim((basis[0], basis[-1]))
    ax.invert_yaxis()

    ax.set_xlabel(grid.ANGLE_NAME)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_trace(trace: grid.BaseTrace,
               figsize: Tuple[int, int] = (3, 5),
               fig_axes: tuple = None,
               plot_params: dict = None
               ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {}

    ax.plot(trace.values, trace.basis, **plot_params)
    ax.set_xlabel(trace.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()

    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_prestack_trace(trace: grid.BasePrestackTrace,
                        scaling: float = 1.0,
                        figsize: Tuple[int, int] = (3, 5),
                        fig_axes: tuple = None,
                        plot_params: dict = None
                        ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {'color': 'b'}

    for theta in trace.angles:
        ax.plot(scaling*trace[theta].values +
                theta, trace.basis, **plot_params)

    fig.suptitle(trace.name)

    ax.set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()

    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_prestack_wiggle_trace(trace: grid.BasePrestackTrace,
                               figsize: Tuple[int, int] = (6, 6),
                               scaling: float = 10.0,
                               decimate_every_n: int = 1,
                               fig_axes: tuple = None
                               ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig, ax = fig_axes

    for theta in trace.angles[::decimate_every_n]:

        _x = scaling*trace[theta].values + theta
        ax.plot(_x, trace.basis, color='k', lw=0.5)
        ax.plot(theta*np.ones_like(_x), trace.basis, color='k', lw=0.2)

        ax.fill_betweenx(trace.basis, theta, _x, where=(_x >= theta),
                         color='b', alpha=.6, interpolate=True)
        ax.fill_betweenx(trace.basis, theta, _x, where=(_x < theta),
                         color='r', alpha=.6, interpolate=True)

    fig.suptitle(trace.name)
    ax.set_xlabel(grid.ANGLE_NAME)
    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    #_space = 0.05*np.abs(trace.values.max())
    #ax.set_xlim((trace.values.min() - _space, (trace.values.max() + scaling + _space)))
    # ax.set_xticks([])

    if trace.is_twt:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_wiggle_trace(trace: grid.BaseTrace,
                      figsize: Tuple[int, int] = (4, 4),
                      repeat_n_times: int = 7,
                      scaling: float = 1.0,
                      fig_axes: tuple = None
                      ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig, ax = fig_axes

    for offset in np.linspace(0.0, scaling*1.0, num=repeat_n_times):

        if repeat_n_times == 1:
            # trace in the middle
            offset = scaling / 2.0

        _x = trace.values + offset
        ax.plot(_x, trace.basis, color='k', lw=0.5)
        ax.plot(offset*np.ones_like(_x), trace.basis, color='k', lw=0.2)

        ax.fill_betweenx(trace.basis, offset, _x, where=(_x >= offset),
                         color='b', alpha=.6, interpolate=True)
        ax.fill_betweenx(trace.basis, offset, _x, where=(_x < offset),
                         color='r', alpha=.6, interpolate=True)

    ax.set_xlabel(trace.name)
    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    _space = 0.05*np.abs(trace.values.max())
    ax.set_xlim((trace.values.min() - _space,
                (trace.values.max() + scaling + _space)))
    ax.set_xticks([])

    if trace.is_twt:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_trace_as_pixels(trace: grid.BaseTrace,
                         repeat_n_times: int = 16,
                         figsize: Tuple[int, int] = (3, 4),
                         wiggle_scale: float = 2.0,
                         fig_axes: tuple = None,
                         im_params: dict = None
                         ) -> plt.subplots:

    absmax = np.abs(trace.values).max()

    pixels = np.empty((repeat_n_times, trace.size))
    for i in range(repeat_n_times):
        pixels[i, :] = trace.values

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    extent = [0, repeat_n_times, trace.basis[0], trace.basis[-1]]

    if im_params is None:
        im_params = dict(cmap='RdYlBu_r', vmin=-.85, vmax=.85)

    im = ax.imshow(pixels.T, interpolation='bilinear',
                   aspect='auto', extent=extent, **im_params)

    ax.plot((repeat_n_times//2)*np.ones_like(trace.values)
            + wiggle_scale*(trace.values / absmax), trace.basis,
            color='k')

    ax.plot((repeat_n_times//2)*np.ones_like(trace.values), trace.basis,
            color='k', lw=0.2, alpha=0.8)

    ax.set_xlabel(trace.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)
    ax.set_xticks([])
    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    cbar = fig.colorbar(im, ax=ax, shrink=1.0, orientation='horizontal',
                        pad=0.02, aspect=5)
    cbar.set_label("", fontsize=11, rotation=90, y=.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_prestack_trace_as_pixels(trace: grid.BasePrestackTrace,
                                  figsize: Tuple[int, int] = (4, 5),
                                  wiggle_scale: float = 2.0,
                                  fig_axes: tuple = None,
                                  im_params: dict = None,
                                  decimate_wiggles: int = 1
                                  ) -> plt.subplots:

    absmax = np.abs(trace.values).max()

    pixels = trace.values

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    extent = [trace.angles[0], trace.angles[-1],
              trace.basis[0], trace.basis[-1]]

    if im_params is None:
        im_params = dict(cmap='RdYlBu_r')

    im = ax.imshow(pixels.T, extent=extent, interpolation='bilinear',
                   aspect='auto', **im_params)

    for theta in trace.angles[::decimate_wiggles]:
        ax.plot(theta*np.ones_like(trace[theta].values)
                + wiggle_scale*(trace[theta].values / absmax), trace.basis,
                color='k', lw=0.5)

    fig.suptitle(trace.name)
    ax.set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)
    # ax.set_xticks([])
    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal',
                        pad=0.15, aspect=20)
    cbar.set_label("", fontsize=11, rotation=90, y=.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_prestack_residual_as_pixels(trace1: grid.BasePrestackTrace,
                                     trace2: grid.BasePrestackTrace,
                                     figsize: Tuple[int, int] = (6, 6),
                                     im_params: dict = None,
                                     ) -> plt.subplots:

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    extent = [trace1.angles[0], trace1.angles[-1],
              trace1.basis[0], trace1.basis[-1]]

    if im_params is None:
        im_params = dict(cmap='RdYlBu_r')

    im = axes[0].imshow(trace1.values.T, extent=extent, interpolation='bilinear',
                        aspect='auto', **im_params)
    axes[1].imshow(trace2.values.T, extent=extent, interpolation='bilinear',
                   aspect='auto', **im_params)

    # residual
    residual = []
    for theta in trace1.angles:
        residual.append(grid.Seismic(trace1[theta].values - trace2[theta].values,
                                     trace1.basis, 'twt', name='Residual', theta=theta))
    residual = grid.PreStackSeismic(residual, name='Residual')

    axes[2].imshow(residual.values.T, extent=extent, interpolation='bilinear',
                   aspect='auto', **im_params)

    axes[0].set_title(trace1.name)
    axes[1].set_title(trace2.name)
    axes[2].set_title(residual.name)

    axes[0].set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
    axes[0].set_ylabel(trace1.basis_type)
    # ax.set_xticks([])
    if trace1.is_twt or trace1.is_tlag:
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    for ax in axes[1:]:
        ax.set_yticklabels("")
        ax.set_xticklabels("")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.6, orientation='vertical')

    cbar.set_label("", fontsize=11, rotation=90, y=.5)

    # fig.tight_layout()

    return fig, axes, cbar


def plot_dynamic_xcorr(dxcorr: grid.DynamicXCorr,
                       figsize: Tuple[int, int] = (4, 5),
                       fig_axes: tuple = None,
                       im_params: dict = None
                       ) -> plt.subplots:

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    # basis in seconds, lags in ms
    extent = [1000*dxcorr.lags_basis[0], 1000*dxcorr.lags_basis[-1],
              dxcorr.basis[-1], dxcorr.basis[0]]

    if im_params is None:
        im_params = dict(cmap='RdYlBu_r', vmin=-.85, vmax=.85)

    im = ax.imshow(dxcorr.values, extent=extent, interpolation='bilinear',
                   aspect='auto', **im_params)

    # fig.suptitle(dxcorr.name)
    ax.set_xlabel(dxcorr.name)
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel(dxcorr.lag_type)
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(dxcorr.basis_type)
    # ax.set_xticks([])
    if dxcorr.is_twt or dxcorr.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, orientation='vertical',
                        pad=0.05, aspect=30)
    cbar.set_label("", fontsize=11, rotation=90, y=.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_optimization_objective(ax_client, fig_axes: tuple = None, figsize=(6, 3)):

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    df = ax_client.get_trials_data_frame().sort_values('trial_index')
    tidx = df['trial_index']
    ax.plot(tidx, df['goodness_of_match'])

    ax.set_title("Optimization objective")
    ax.set_xlabel("Iteration #")
    ax.set_ylabel("Central correlation")
    ax.set_xlim([tidx.values[0], tidx.values[-1]])

    fig.tight_layout()

    return fig, ax

#######################
# warping
#######################


def plot_warping(ref_trace: grid.BaseTrace,
                 other_trace: grid.BaseTrace,
                 lags: grid.DynamicLag,
                 scale: float = 1.0,
                 figsize: Tuple[int, int] = (6.5, 4.5),
                 fig_axes: tuple = None
                 ) -> plt.subplots:
    """Plot the optimal warping between to sequences.
    ref_trace: real seismic
    other_trace: synthetic seismic

    :param s1: From sequence. (real seismic)
    :param s2: To sequence. (synth seismic)
    :param lags: Index time lags.

    Original code:
    https://github.com/wannesm/dtaidistance/dtaidistance/dtw_visualisation.py
    """
    assert ref_trace.basis_type == other_trace.basis_type
    assert ref_trace.basis_type == lags.basis_type

    s1 = np.copy(ref_trace.values)
    s2 = np.copy(other_trace.values)

    s1 /= np.abs(s1).max()
    s1 += scale

    s2 /= np.abs(s2).max()
    s2 -= scale

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(ref_trace.basis, s1, color='royalblue')
    ax.plot(other_trace.basis, s2, color='royalblue')
    ax.set_yticks([])
    ax.set_xlabel(ref_trace.basis_type)

    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}

    lags_idx = np.round(lags.values / lags.sampling_rate).astype(np.int)
    path = [(i, i+lag) for i, lag in enumerate(lags_idx)]

    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        if r_c >= s1.size or c_c >= s1.size:
            continue

        r_c_value = ref_trace.basis[r_c]
        c_c_value = ref_trace.basis[c_c]

        con = ConnectionPatch(xyA=[r_c_value, s1[r_c]], coordsA=ax.transData,
                              xyB=[c_c_value, s2[c_c]], coordsB=ax.transData,
                              **line_options)
        lines.append(con)

    for line in lines:
        fig.add_artist(line)

    fig.suptitle("Warping Path")
    fig.tight_layout()

    return fig, ax
