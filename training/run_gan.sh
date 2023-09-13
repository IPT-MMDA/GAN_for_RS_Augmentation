#!/bin/bash

ROOT_DIR=$(cd ../utils && pwd)
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR/utils

JOB_DIR="data"
python3 gan_training.py --job-dir $JOB_DIR
