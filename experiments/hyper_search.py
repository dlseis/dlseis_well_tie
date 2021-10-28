# File must be started with ../hyper_search.sh

print("Loading modules...")

import sys, yaml, time, copy

from pathlib import Path
from tqdm import tqdm

from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models as ax_Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy


from wtie.dataset import BaseDataset
from wtie.learning.model import LightModel, LightVariationalModel
from wtie.utils.logger import Logger, display_time_from_seconds
from wtie.modeling.utils import create_or_load_synthetic_dataset





WORKDIR = Path().absolute() # CAREFULL NEEDS CHECK
print("Work directory: %s" % WORKDIR)

with open('./parameters.yaml', 'r') as yaml_file:
    parameters = yaml.load(yaml_file, Loader=yaml.SafeLoader)




SAVE_DIR = WORKDIR





## PARAMETERS & SEARCH SPACE
with open('./search_space.yaml', 'r') as yaml_file:
    _params = yaml.load(yaml_file, Loader=yaml.SafeLoader)

NUM_SAMPLING = _params['meta']['num_sampling']
NUM_TRAINING_SAMPLES = _params['meta']['num_training_samples']

SEARCH_SPACE = [v for k,v in _params.items() if k != 'meta']


## MAIN
def main():
    # init
    start_time = time.time()
    logger = Logger(WORKDIR / 'search.log')


    # hyper param optim
    logger.write("\nSTART SEARCH...", highlight=True)
    search()


    # log
    logger.write("\nTOTAL TIME:", highlight=True)
    logger.write(display_time_from_seconds(time.time() - start_time))
    logger.write("\nDONE.", highlight=True)


## TRAINING
def prepare_dataset(save_dir):
    start_time = time.time()

    # log
    logger = Logger(WORKDIR / 'search.log')




    # files
    h5_file = save_dir / 'dataset.h5'
    h5_group_training = 'synth01/training'
    h5_group_validation = 'synth01/validation'

    # params
    params = copy.deepcopy(parameters['synthetic_dataset'])
    params['num_training_samples'] = NUM_TRAINING_SAMPLES

    logger.write("Create or load dataset...")
    train_creator, val_creator = create_or_load_synthetic_dataset(h5_file,
                                                                  h5_group_training,
                                                                  h5_group_validation,
                                                                  params
                                                                  )

    base_train_dataset = BaseDataset(data_creator=train_creator)
    base_val_dataset = BaseDataset(data_creator=val_creator)

    logger.write("Dataset creation/loading time:")
    logger.write(display_time_from_seconds(time.time() - start_time))

    return base_train_dataset, base_val_dataset



def search():
    start_time = time.time()

    # log
    logger = Logger(WORKDIR / 'search.log')

    # dataset
    # save directory
    save_dir = SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)




    # train/val data
    base_train_dataset, base_val_dataset = prepare_dataset(save_dir)



    # search strategy
    logger.write("Start search...")

    n_sobol = int(0.5*NUM_SAMPLING) #50%
    n_bayes = NUM_SAMPLING - n_sobol #50%

    ax_gen_startegy = GenerationStrategy(
        [GenerationStep(ax_Models.SOBOL, num_trials=n_sobol),
         GenerationStep(ax_Models.BOTORCH,num_trials=n_bayes)
         ]
        )


    ax_client = AxClient(generation_strategy=ax_gen_startegy,
                         verbose_logging=False)

    ax_client.create_experiment(
        name="variational_wavelet_estimation",
        parameters=SEARCH_SPACE,
        objective_name="reconstruction_error",
        minimize=True,
        choose_generation_strategy_kwargs=None
        )


    for i in tqdm(range(NUM_SAMPLING)):
        h_params, trial_index = ax_client.get_next_trial()

        # overight default params, a bit ugly...
        current_params = copy.deepcopy(parameters['network'])
        network_kwargs = dict()
        for key, value in h_params.items():
            if key in ['kernel_size', 'downsampling_factor', 'dropout_rate']:
                network_kwargs[key] = value
            else:
                current_params[key] = value
        current_params['network_kwargs'] = network_kwargs

        # test
        for k,v in current_params.items():
            if k in h_params:
                assert v == h_params[k]

        learner = LightVariationalModel(base_train_dataset,
                                               base_val_dataset,
                                               current_params)

        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=learner.train_and_validate())






    ax_client.save_to_json_file(save_dir / 'search_results.json')

    logger.write("Total search time:")
    logger.write(display_time_from_seconds(time.time() - start_time))





if __name__ == '__main__':
    sys.exit(main())
