# File must be started with ../main.sh

print("Loading modules...")

import sys, yaml, h5py, time

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


from wtie.dataset import BaseDataset
from wtie.learning.model import Model, VariationalModel
from wtie.utils.logger import Logger, display_time_from_seconds


from wtie.modeling.utils import create_or_load_synthetic_dataset





WORKDIR = Path().absolute() # CAREFULL NEEDS CHECK
print("Work directory: %s" % WORKDIR)

with open('./parameters.yaml', 'r') as yaml_file:
    parameters = yaml.load(yaml_file, Loader=yaml.SafeLoader)




TRAINING_SAVE_DIR = WORKDIR / "training"
EVALUATION_SAVE_DIR = WORKDIR / "evaluation"


## MAIN
def main():
    # init
    start_time = time.time()
    logger = Logger(WORKDIR / 'experiment.log')


    # training
    logger.write("\nSTART EXPERIMENT...", highlight=True)
    train()


    # log
    logger.write("\nTOTAL TIME:", highlight=True)
    logger.write(display_time_from_seconds(time.time() - start_time))
    logger.write("\nDONE.", highlight=True)


## TRAINING




def train():
    start_time = time.time()

    # log
    logger = Logger(WORKDIR / 'experiment.log')

    # dataset
    # save directory
    save_dir = TRAINING_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=False)

    # files
    h5_file = TRAINING_SAVE_DIR / 'dataset.h5'
    h5_group_training = 'synth01/training'
    h5_group_validation = 'synth01/validation'

    logger.write("Create or load dataset...")
    training_creator, validation_creator = create_or_load_synthetic_dataset(h5_file,
                                                                            h5_group_training,
                                                                            h5_group_validation,
                                                                            parameters['synthetic_dataset']
                                                                            )
    logger.write("Dataset creation/loading time:")
    logger.write(display_time_from_seconds(time.time() - start_time))

    # params
    params = parameters['network']

    # reset for measuring training time
    start_time = time.time()

    # train/val data
    logger.write("Instanciate train/val base datasets")
    base_train_dataset = BaseDataset(data_creator=training_creator)
    base_val_dataset = BaseDataset(data_creator=validation_creator)



    # tensorboard
    if params['tensorboard'] is True:
        tensorboard = SummaryWriter(log_dir= save_dir / 'tensorboard')
        logger.write("Monitoring training with Tensorboard")
    else:
        tensorboard = None



    # model
    logger.write("Instanciate model")
    learner = VariationalModel(save_dir,
                                      base_train_dataset,
                                      base_val_dataset,
                                      params,
                                      logger,
                                      tensorboard=tensorboard
                                      )



    # training
    learner.train()

    # tb
    if tensorboard is not None:
        tensorboard.close()

    logger.write("Total training time:")
    logger.write(display_time_from_seconds(time.time() - start_time))





if __name__ == '__main__':
    sys.exit(main())
