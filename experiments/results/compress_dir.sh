#!/bin/bash

# Requires a single argument which will be the name of the  directory to compress



# experient name
if [ $# -eq 0 ]
  then
    echo "please provide name of experiment\t[aborting]"
    exit 1
fi

DIR_NAME="${1}"
DIR_NAME="${DIR_NAME///}" #remove / from name if any

#echo "$DIR_NAME"

# archive name
ARC_NAME=""${DIR_NAME}".tar.gz"

#echo "$ARC_NAME"

# clean
cd "$DIR_NAME"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name .pytest_cache -type d -exec rm -rf {} \;
cd ..

# compress
tar -czvf "${DIR_NAME}.tar.gz" $DIR_NAME




