#!/bin/bash
kernels=(
  "bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds"
  "scheduler_bkdstprcmp_shuffle_trimotm_ds"
  "scheduler_bkdstprcmp_xdovlpunroll_shuffle_ds"
  "scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm"
  "scheduler_bkdstprcmp_xdovlpunroll_vtile_trimotm_ds"
  "scheduler_xdovlpunroll_shuffle_trimotm_ds"
)

for kernel in "${kernels[@]}"; do
  echo "Profiling $kernel"
  srun --pty -A dphpc -t 60 ncu --launch-count 2 --launch-skip 2 \
    --kernel-name "compute_next_layers_kernel_batch_${kernel}" \
    -o "res/profile_${kernel}" -f \
    ../build/bin/pricing_cli batch-random-benchmark --n-random-runs 100 -n 20000 --skip-sanity-checks --filter-by-name "${kernel}"
    cp "../src/backends/cuda/vanilla_american_binomial_cuda/${kernel}.cu" res/
done
