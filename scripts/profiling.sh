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
echo '=== EXTRACTING MATCHING KERNELS ==='
MATCHING_KERNELS=\$(nsys stats ../profile_res/profile_report.sqlite \
  --report cuda_gpu_kern_sum --format csv 2>/dev/null | \
  tail -n +2 | awk -F'\"' 'NF>=2 {print \$(NF-1)}' | grep -v '^$' | grep -E \"$REGEX_KERNEL\" | sort -u)

echo \"Found matching kernels:\"
echo \"\$MATCHING_KERNELS\"

echo ''
echo '=== NSIGHT COMPUTE PROFILING ==='
KERNEL_NUM=0
while IFS= read -r KERNEL; do
  [ -z \"\$KERNEL\" ] && continue
  KERNEL_NUM=\$((KERNEL_NUM + 1))

  # Extract just the kernel name without return type, template params, or function params
  # For non-templated: 'kernel_name(...)' -> 'kernel_name'
  # For templated: 'void kernel_name<T>(...)' -> 'kernel_name'
  KERNEL_NAME=\$(echo \"\$KERNEL\" | sed 's/^void //; s/<.*//; s/(.*//')

  echo \"\"
  echo \"Profiling kernel \$KERNEL_NUM: \$KERNEL_NAME\"

  ncu --target-processes all --kernel-name \"\$KERNEL_NAME\" --set full \
    --launch-skip 10 --launch-count 5 -f \
    -o ../profile_res/profile_kernel_\${KERNEL_NUM} \
    ./bin/pricing_cli benchmark --parameters $PARAMETER --filter-by-name $FUNCTION_NAME
done <<< \"\$MATCHING_KERNELS\"

echo ''
echo '=== PROFILING COMPLETE ==='
echo \"Generated profile files:\"
ls -lh ../profile_res/profile_kernel_*.ncu-rep 2>/dev/null || echo \"No kernel profiles generated\"
"
