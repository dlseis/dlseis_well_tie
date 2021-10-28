"""Utils function using the tool Ax to perform global optimization"""

import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy



def create_ax_client(num_iters: int,
                     random_ratio: float=0.6,
                     verbose: bool=False,
                     **kwargs) -> AxClient:
    """Default client, first sobol, then bayes. """
    n_sobol = int(random_ratio*num_iters)
    n_bayes = num_iters - n_sobol

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ax_gen_startegy = GenerationStrategy(
        [GenerationStep(Models.SOBOL, num_trials=n_sobol),
         GenerationStep(Models.BOTORCH,num_trials=n_bayes,
                        max_parallelism=3,
                        model_kwargs = {"torch_dtype": torch.float,
                                        "torch_device": device}
                        )
         ]
        )

    return AxClient(generation_strategy=ax_gen_startegy,
                    verbose_logging=verbose, **kwargs)