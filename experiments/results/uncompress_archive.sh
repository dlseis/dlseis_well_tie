#!/bin/bash

# Requires a single argument which will be the name of the  archive to uncompress



# experient name
if [ $# -eq 0 ]
  then
    echo "please provide name of experiment\t[aborting]"
    exit 1
fi

ARC_NAME="${1}"

tar -xzvf $ARC_NAME

