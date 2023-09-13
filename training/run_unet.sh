#!/bin/bash

ROOT_DIR=$(cd ../utils && pwd)
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR/utils

JOB_DIR="data"
TYPE="real"
# TYPE="gan"
# TYPE="replace"
# TYPE="stat"

python3 unet_training.py --job-dir "$JOB_DIR" --type "$TYPE"
