#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -A bb1097
#SBATCH -n 3
#SBATCH --mem=128000
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/job_%j.log

# Create log directory
mkdir -p slurm_logs

# Run Python with unbuffered output
cd /home/b/b383521/Gravity_Wave_Machine_Learning_Pramterzation
/home/b/b383521/venvs/py310_jupyter/bin/python -m machine_learning_codes.model.model