#!/bin/bash

echo "Running tests..."

# main directory
STARTDIR=$(pwd)

# test directory
TESTDIR="${STARTDIR}/tests"
TMPDIR="${TESTDIR}/tmp"

# start
cd "${TESTDIR}"
pytest -v --basetemp=${TMPDIR}
# pytest -v --basetemp=${TMPDIR} -k model
