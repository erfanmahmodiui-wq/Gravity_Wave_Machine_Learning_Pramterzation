#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -A bb1097
#SBATCH -n 2
#SBATCH --mem=300000
#SBATCH -t 05:00:00
#SBATCH -o slurm_logs/job_%A_%a.log
#SBATCH --array=0

mkdir -p slurm_logs

echo "Running file_slide $SLURM_ARRAY_TASK_ID"
/home/b/b383521/venvs/py310_jupyter/bin/python -u \
    /work/bb1097/b383521/EOF_codes/avrage_slide.py $SLURM_ARRAY_TASK_ID