#!/usr/bin/env bash
set -euo pipefail

account="$1"
partition="boost_usr_prod"
cluster_name="leonardo"

# nodes=(1 2 4 8 16 32 64)
nodes=(2)

gpus_per_node=4
tasks_per_node=4
cpus_per_task=4

# channels=(2 4 8 16 32)
# protocols=(LL LL128 Simple)
# threads=(64 128 256 512)

channels=(32)
protocols=(Simple)
threads=(512)

for node in "${nodes[@]}"; do
  for channel in "${channels[@]}"; do
    for protocol in "${protocols[@]}"; do
      for thread in "${threads[@]}"; do
        sbatchman configure slurm \
          --name "a2a_${node}_${channel}_${protocol}_${thread}" \
          --cluster-name "${cluster_name}" \
          --partition "${partition}" \
          --account "${account}" \
          --nodes "${node}" \
          --tasks-per-node "${tasks_per_node}" \
          --cpus-per-task "${cpus_per_task}" \
          --gpus "${gpus_per_node}" \
          --exclusive \
          --env "NCCL_PROTO=${protocol}" \
          --env "NCCL_MIN_CTAS=${channel}" \
          --env "NCCL_MAX_CTAS=${channel}" \
          --env "NCCL_NTHREADS=${thread}"
      done
    done
  done
done
