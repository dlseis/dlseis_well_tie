#!/bin/bash

echo "Create wtie conda environment"

# https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script
#source /opt/anaconda3/etc/profile.d/conda.sh
source $(conda info --base)/etc/profile.d/conda.sh

conda env create -f environment_linux.yml
