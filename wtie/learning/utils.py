import warnings

import math
import numpy as np

from wtie.learning.losses import VariationalLoss

class EarlyStopping:
    """Original implementation:
        https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self,
                 mode:str='min',
                 min_delta:float=0,
                 patience: int=10,
                 percentage: bool=True,
                 min_epochs: int=None,
                 alpha:float=.1):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.current_epoch = 0
        self.min_epochs = min_epochs
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if alpha is not None:
            self.averager = RunningAverage(alpha=alpha)
        else:
            self.averager = None


        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics: float) -> bool:
        self.current_epoch += 1

        if self.averager is not None:
            metrics = self.averager(metrics)

        if self.best is None:
            self.best = metrics
            return False

        if self.current_epoch < self.min_epochs:
            return False

        if np.isnan(metrics):
            warnings.warn("Early stopping got NAN as metric.")
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)



class RunningAverage:
    def __init__(self, alpha: float):
        if not 0 < alpha < 1:
            raise ValueError("out of range, alpha=%f" % alpha)
        self.alpha = alpha
        self.x_old = None

    def __call__(self, x: float) -> float:
        if self.x_old is None:
            self.x_old = x
            return x

        x_new = self.x_old + (x - self.x_old)*self.alpha
        self.x_old = x_new
        return x_new



class AlphaScheduler:
    def __init__(self,
                 loss: VariationalLoss,
                 alpha_init: float,
                 alpha_max: float,
                 rate: float,
                 every_n_epoch:int):

        self.loss = loss

        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.rate = rate
        self.every_n_epoch = every_n_epoch

        self.current_epoch = 1

    @property
    def alpha(self):
        current_rate = math.pow(self.rate,
                                math.floor(self.current_epoch / self.every_n_epoch))
        return min(self.alpha_init * current_rate, self.alpha_max)

    def step(self):
        self.current_epoch = self.current_epoch + 1
        # update internal state of loss class
        # valid outside of this class as well
        self.loss.alpha = self.alpha