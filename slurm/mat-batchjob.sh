#!/bin/bash
#SBATCH --ntasks=1                      # Number of tasks (see below)
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8               # Number of CPU coes per task
#SBATCH --mem=48G                       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes=1                       # Ensure that all cores are on one machine
#SBATCH --time=1-00:00                  # Runtime in D-HH:MM
#SBATCH -o logs/cblearn-benchmark_%A_%a.out # Standard output - make sure this is not on $HOME
#SBATCH -e logs/cblearn-benchmark_%A_%a.err # Standard error - make sure this is not on $HOME
#SBATCH --mail-type=FAIL                # Type of email notification - BEGIN, END, FAIL, ALL 
#SBATCH --mail-user=david-elias.kuenstle@uni-tuebingen.de #Email to which notificatibns will be sent  


# print info about current job
scontrol show job $SLURM_JOB_ID
#commands
singularity run --bind ${PWD}:/home/docker --pwd /home/docker --env MLM_LICENSE_FILE=27000@matlab-campus.uni-tuebingen.de docker://mathworks/matlab:r2022a sh -c "$*"