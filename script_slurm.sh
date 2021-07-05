#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
echo "Activating CUDA"
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100
echo "Running python main script into out/experiment.log"
python3 /vol/bitbucket/${USER}/FederatedMedical/main.py > out/experiment.log
echo $! > out/lastExperimentPID.txt
uptime
