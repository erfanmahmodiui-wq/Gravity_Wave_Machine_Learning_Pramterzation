#!/bin/bash
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH -A bb1097
#SBATCH -n 2
#SBATCH --mem=400000
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/job_%j.log

# Create log directory
mkdir -p slurm_logs

# Run Python with unbuffered output
/home/b/b383521/venvs/py310_jupyter/bin/python -u /work/bb1097/b383521/EOF_codes/fluctuation.py