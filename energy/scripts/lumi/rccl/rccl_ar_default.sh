#!/bin/bash
#SBATCH --job-name=rccl_ar_default
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --account=project_465002469
#SBATCH --output=/users/carpenti/energy-ws/Blink-GPU/energy/logs/%x.%j.out
#SBATCH --exclusive


source /users/carpenti/energy-ws/Blink-GPU/energy/scripts/lumi/env/rccl_env.sh
export rccl_DEBUG=INFO     # Very detailed logs (verbose)
export rccl_DEBUG_SUBSYS=GRAPH,COLL,INIT

### HIP RUN ###
# ar
srun  /users/carpenti/energy-ws/Blink-GPU/energy/build/bin/ar_rccl /users/carpenti/energy-ws/Blink-GPU/energy/logs/ar_rccl.csv