## Configure CMake to include debug symbols (RelWithDebInfo) and CUDA line-info
## - RelWithDebInfo keeps optimizations but adds debug info (good for profiling)
## - Add `-G` to CMAKE_CUDA_FLAGS if you need device debugging (slower) instead
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_CUDA_COMPILER=/cluster/data/cuda/13.0.2/bin/nvcc \
	-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
	-DCMAKE_CUDA_FLAGS="-lineinfo -g" ..
