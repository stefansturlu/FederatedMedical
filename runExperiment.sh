#!/bin/bash

source ./my_env/bin/activate
nohup python -u main.py > out/experiment.log 2>&1 &
echo $! > out/lastExperimentPID.txt

