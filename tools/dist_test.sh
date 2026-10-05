#!/usr/bin/env bash

source $HOME/anaconda3/bin/activate pyskl1

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# Print PYTHONPATH for debugging
echo "PYTHONPATH: $PYTHONPATH"

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:4}
