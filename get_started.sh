#!/usr/bin/env bash
CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$CODE_DIR

DATA_DIR=data
DOWNLOAD_DIR=download

python2 $CODE_DIR/preprocessing/dwr.py

# Data processing for TensorFlow
python2 $CODE_DIR/qa_data.py --glove_dim 100
