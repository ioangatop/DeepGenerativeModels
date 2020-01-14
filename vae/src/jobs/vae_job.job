#!/bin/bash
#SBATCH --job-name=vae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
# SBATCH -C Titan
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igatopou/projects/vae/src/jobs/out/slurm-%j.out

module load cuda80/toolkit prun
module load opencl-nvidia/8.0

for PRIOR in 'mog'
do
    srun python -u /home/igatopou/projects/vae/main.py --prior $PRIOR
done
