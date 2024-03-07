#!/usr/bin/bash

# run this script with the following command:
# torchrun --no-python --nnodes 1 --nproc-per-node 2 ./run.sh

# gets the directory containing this script, cd, then runs hello.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR &&
python3 hello.py
