"""Small collections to facilitate the manipulation of the input/output
of the wtie.optimize.tie functions. """

import matplotlib.pyplot as plt

#from collections import namedtuple
from dataclasses import dataclass

from ax.service.ax_client import AxClient

from wtie import grid
from wtie.utils import viz as _viz




#InputSet = namedtuple('InputSet',('logset_md','seismic', 'wellpath', 'table'))

@dataclass
class InputSet:
    """ """

    logset_md: grid.LogSet
    seismic: grid.seismic_t
    wellpath: grid.WellPath
    table: grid.TimeDepthTable


    def __post_init__(self):
        assert self.logset_md.is_md



    def plot_inputs(self, figsize: tuple=(9,4), scale: float=1.0):
        fig,axes = plt.subplots(1,4,gridspec_kw={'width_ratios': [1,3,3,2]},
                                figsize=figsize)
        _viz.plot_trace(self.logset_md.AI, fig_axes=(fig,axes[0]))
        _viz.plot_wellpath(self.wellpath, fig_axes=(fig,axes[1]))
        _viz.plot_td_table(self.table, fig_axes=(fig,axes[2]))

        if self.seismic.is_prestack:
            _viz.plot_prestack_trace(self.seismic, scale=scale, fig_axes=(fig,axes[3]))
        else:
            _viz.plot_trace(self.seismic, fig_axes=(fig,axes[3]))

        return fig, axes




@dataclass
class OutputSet:
    """ """
    wavelet: grid.wlt_t
    logset_twt: grid.LogSet
    seismic: grid.seismic_t
    synth_seismic: grid.seismic_t
    wellpath: grid.WellPath
    table: grid.TimeDepthTable
    r: grid.ref_t

    xcorr: grid.xcorr_t = None
    dlags: grid.DynamicLag = None
    dxcorr: grid.DynamicXCorr = None
    ax_client: AxClient=None


    def plot_tie_window(self,
                        wiggle_scale: float=None,
                        figsize: tuple=(9,5),
                        **kwargs
                        ) -> plt.subplots:

        if self.seismic.is_prestack:
            fig, axes = _viz.plot_prestack_tie_window(self.logset_twt,
                                                      self.r,
                                                      self.synth_seismic,
                                                      self.seismic,
                                                      self.xcorr,
                                                      figsize=figsize,
                                                      wiggle_scale_syn=wiggle_scale,
                                                      wiggle_scale_real=wiggle_scale,
                                                      **kwargs)
        else:
            fig, axes = _viz.plot_tie_window(self.logset_twt,
                                             self.r,
                                             self.synth_seismic,
                                             self.seismic,
                                             self.xcorr,
                                             self.dxcorr,
                                             figsize=figsize,
                                             wiggle_scale_syn=wiggle_scale,
                                             wiggle_scale_real=wiggle_scale,
                                             **kwargs
                                             )
        axes[0].locator_params(axis='y', nbins=16)

        return fig, axes

    def plot_wavelet(self, **kwargs):
        if self.wavelet.is_prestack:
            return _viz.plot_prestack_wavelet(self.wavelet, **kwargs)
        else:
            return _viz.plot_wavelet(self.wavelet, **kwargs)

    def plot_optimization_objective(self, **kwargs):
        return _viz.plot_optimization_objective(self.ax_client,
                                                **kwargs)
