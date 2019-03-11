#!/bin/bash

module load cuda/9.0.176
sleep 0`bc -l <<< "scale=4 ; ${RANDOM}/32767"`s
echo $(pwd)

python3 scripts/stardist/master.py
