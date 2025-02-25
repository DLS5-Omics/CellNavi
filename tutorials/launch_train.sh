# !/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"


if [ -z $WORLD_SIZE ]; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

  torchrun --nproc_per_node=${GPU_COUNT} \
  	   $HERE/start_train.py
else
  python $HERE/start_train.py
fi