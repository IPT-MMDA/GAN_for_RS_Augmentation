#!/bin/bash

JOB_DIR="data"
TYPE="real"
# TYPE="gan"
# TYPE="replace"
# TYPE="stat"

python3 unet_training.py --job-dir "$JOB_DIR" --type "$TYPE"
