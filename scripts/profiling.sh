# PARAMETERS

# default parameter hard dataset
REGEX_KERNEL="${1}"
FUNCTION_NAME="${2:-cuda}"
PARAMETER="${3:-cuda_debug}"
echo "Profiling with parameters set: ${PARAMETER}"
echo "And kernel regex: ${REGEX_KERNEL}"

## BUILD
cd ./build || exit 1
bash ../scripts/compile_on_cluster.sh || exit 1
make -j 4 || exit 1

## PROFILE

# ---- GPU Allocation for Profiling (warmup + nsys + ncu in single allocation)

echo ""
echo "=== PROFILING WITH WARMUP (Single GPU Allocation) ==="
srun --pty -A dphpc -t 60 bash -c "
set -e

# Initialize module system and load CUDA
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module add cuda/13.0.2

echo ''
echo '=== NCU VERSION ==='
ncu --version

echo ''
echo '=== NCU SUPPORTED CHIPS ==='
ncu --list-chips

echo ''
echo '=== GPU INFO ==='
nvidia-smi

echo ''
echo '=== WARMUP RUN ==='
./bin/pricing_cli benchmark --parameters $PARAMETER --filter-by-name $FUNCTION_NAME 
echo 'Warmup complete'

echo ''
echo '=== NSIGHT SYSTEMS PROFILING ==='
/usr/local/bin/nsys profile --stats=true -o ../profile_res/profile_report -f true \
  ./bin/pricing_cli benchmark --parameters $PARAMETER --filter-by-name $FUNCTION_NAME

echo ''
echo '=== NSIGHT COMPUTE PROFILING ==='
ncu --target-processes all --kernel-name \"regex:$REGEX_KERNEL\" --set full \
  --launch-skip 100 --launch-count 5 -f -o ../profile_res/profile_kernel \
  ./bin/pricing_cli benchmark --parameters $PARAMETER --filter-by-name $FUNCTION_NAME
"
