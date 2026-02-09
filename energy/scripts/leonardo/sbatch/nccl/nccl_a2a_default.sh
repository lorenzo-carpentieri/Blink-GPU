#!/bin/bash
#SBATCH --job-name=nccl_a2a_default
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=000:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_OMG-25
#SBATCH --output=/leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/logs/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


source /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/scripts/leonardo/env/set_nccl_env.sh
export NCCL_DEBUG=INFO     # Very detailed logs (verbose)
export NCCL_DEBUG_SUBSYS=GRAPH,COLL,INIT

### CUDA RUN ###
# a2a
srun  /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/build/bin/a2a_nccl /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/logs/a2a.csv