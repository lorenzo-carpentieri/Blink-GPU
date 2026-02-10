
sbatchman launch \
    --config a2a_2_32_Simple_512 \
    --tag a2a_2_32_Simple_512 \
    "source /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/scripts/leonardo/env/set_nccl_env.sh && srun /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/build/bin/a2a_nccl /leonardo/home/userexternal/kfan0000/mpi-energy/Blink-GPU/energy/logs/a2a.csv"