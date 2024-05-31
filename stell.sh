#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores 
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --gpus=a100:1
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest
source activate cuda
python train.py --epochs 1 --lr 0.001 --network resnet152