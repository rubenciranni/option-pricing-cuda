# PARAMETERS

FUNCTION_NAME="${1:-cuda}"

## BUILD

module load cuda/12.8
cd ./build || exit 1
bash ../scripts/compile_on_cluster.sh || exit 1
make -j 4 || exit 1

## PROFILE
srun --pty -A dphpc -t 60 \
  ncu --target-processes all --nvtx --set full \
    -f -o ../profile_res/profile_kernel \
 ./bin/pricing_cli benchmark --parameters cuda_debug --filter-by-name $FUNCTION_NAME --skip-sanity-checks
# here we run only the cuda implementation benchmarks
