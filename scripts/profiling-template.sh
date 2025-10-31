# echo in red
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${RED}WARNING: This is a template script for profiling. Modify it as needed.${NC}"

# PARAMETERS

# default parameter hard dataset
REGEX_KERNEL="${1}"
FUNCTION_NAME="${2:-cuda}"
PARAMETER="${3:-cuda_debug}"
echo "Profiling with parameters set: ${PARAMETER}"
echo "And kernel regex: ${REGEX_KERNEL}"

## BUILD

module load cuda/12.8
cd ./build || exit 1
bash ../scripts/compile_on_cluster.sh || exit 1
make -j 4 || exit 1

## PROFILE

# ---- Nsight Compute

srun --pty -A dphpc -t 60 \
  nsys profile --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o ../profile_res/profile_report \
 ./bin/pricing_cli benchmark --parameters $PARAMETER   --filter-by-name $FUNCTION_NAME --no-verify
  # here we run only the cuda implementation benchmarks 

# ----- Nsight Systems

# RUn sbatch system
srun --pty -A dphpc -t 60 \
  ncu --kernel-name $REGEX_KERNEL \
    --launch-count 1 -f -o ../profile_res/profile_kernel \
 ./bin/pricing_cli benchmark --parameters $PARAMETER   --filter-by-name $FUNCTION_NAME  --no-verify
# here we run only the cuda implementation benchmarks 