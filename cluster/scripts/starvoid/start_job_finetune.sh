#!/bin/bash
module load cuda/9.0.176
sleep 0`bc -l <<< "scale=4 ; ${RANDOM}/32767"`s
echo $(pwd)

PYTHONPATH=${PYTHONPATH}:/lustre/projects/juglab/StarVoid/StarVoid/noise2seg
export PYTHONPATH
echo $PYTHONPATH
python3 scripts/starvoid/master_finetune.py

