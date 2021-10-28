#!/bin/bash

# This script should be used as the starting point to call well-tie/experiements/main.py
# Requires a single argument which will be the name of the sub-experiement directory





# main directory
STARTDIR=$(pwd)



# experient name
if [ $# -eq 0 ]
  then
    echo "please provide name of experiment\t[aborting]"
    exit 1
fi

EXP_NAME="${1}"


# results directory
RESDIR="${STARTDIR}/experiments/results/${EXP_NAME}"


if [ -d "$RESDIR" ]; then
  # Control will enter here if $RESDIR exists.
  echo "experiment folder already exists\t[aborting]"
  exit 1
fi

echo "Start directory: "${STARTDIR}""
echo "Create new experiment in "${RESDIR}""
mkdir "$RESDIR"



# copy files
echo "Copy source and config files for reproducibility"
cp ./experiments/main.py ${RESDIR}
cp ./experiments/parameters.yaml ${RESDIR}


# start
cd "${RESDIR}"
python main.py