#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sps20
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100
python3 /vol/bitbucket/${USER}/FederatedMedical/main.py > out/experiment.log
echo $! > out/lastExperimentPID.txt
uptime
