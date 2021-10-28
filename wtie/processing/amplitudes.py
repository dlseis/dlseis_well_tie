import numpy as np

normalize = lambda x, a=-1., b=1. : ((b-a) * (x - x.min()) / x.ptp()) + a
