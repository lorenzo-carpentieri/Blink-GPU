#!/bin/bash
#SBATCH --job-name=nccl_ar_default
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


source ./scripts/leonardo/env/set_nccl_env.sh
export NCCL_DEBUG=INFO     # Very detailed logs (verbose)
export NCCL_DEBUG_SUBSYS=GRAPH,COLL,INIT

### CUDA RUN ###
# all_reduce
srun  /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/build/bin/ar_nccl /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/logs/ar_nccl.csv