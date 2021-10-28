#!/bin/bash

echo "Build wtie package in develop mode"

# conda : https://stackoverflow.com/questions/53382383



# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
#CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#($(CONDA_ACTIVATE) wtie ; python setup.py develop)


python setup.py develop