#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

#define IMPL_NAME stprcmp_yunroll_vtile

__global__ void FUNC_NAME(fill_st_buffer_kernel)(double* __restrict__ buffer, const double S,
                                                 const double K, const double u, const int sign,
                                                 const int n) {
    int threadId = (blockIdx.x * blockDim.x + threadIdx.x);  // offset of result idx
    if (threadId > 2 * n) return;
    buffer[threadId] = max(sign * (S * pow(u, threadId - n) - K), 0.0);
}

template <const int OUTPUTS_PER_THREAD>
__global__ void FUNC_NAME(compute_first_layer_kernel)(double* __restrict__ d_option_values,
                                                      double* __restrict__ st_buffer, const int n) {
    int threadId = OUTPUTS_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
    if (threadId > n) return;
#pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        d_option_values[threadId + i] = st_buffer[2 * (threadId + i)];
    }
}

template <const int THREADS_PER_BLOCK, const int OUTPUTS_PER_THREAD>
__global__ void FUNC_NAME(compute_next_layer_kernel)(double* __restrict__ d_option_values,
                                                     double* __restrict__ d_option_values_next,
                                                     double* __restrict__ st_buffer,
                                                     const double prob_up, const double prob_down,
                                                     const int level, const int n) {
    __shared__ double tile[THREADS_PER_BLOCK * OUTPUTS_PER_THREAD + 1];
    int threadGlobalOffset = OUTPUTS_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);

    // 0   1
    // 0 1 2 3
    for (int i = OUTPUTS_PER_THREAD * threadIdx.x; i < THREADS_PER_BLOCK * OUTPUTS_PER_THREAD + 1;
         i += THREADS_PER_BLOCK * OUTPUTS_PER_THREAD) {
        int global_offset = OUTPUTS_PER_THREAD * (blockIdx.x * blockDim.x);
        for (int j = 0; j < OUTPUTS_PER_THREAD; j++)
            tile[i + j] = d_option_values[global_offset + i + j];
    }
    // wait for threads to finish setting up tile
    __syncthreads();

    if (threadGlobalOffset > level) return;
// threadId = (threadId % (level+1));
#pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD && threadGlobalOffset + i <= level; i++) {
        double hold = prob_up * tile[OUTPUTS_PER_THREAD * threadIdx.x + i + 1] +
                      prob_down * tile[OUTPUTS_PER_THREAD * threadIdx.x + i];
        double exercise = st_buffer[2 * (threadGlobalOffset + i) - level + n];
        d_option_values_next[threadGlobalOffset + i] = max(hold, exercise);
    }
}

template <const Hyperparams& h>
double FUNC_NAME(vanilla_american_binomial_cuda)(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {
    const double deltaT = T / n;
    const double u = std::exp(sigma * std::sqrt(deltaT));
    const double d = 1.0 / u;
    const double p = (exp((r - q) * deltaT) - d) / (u - d);
    const double risk_free_rate = std::exp(-r * deltaT);
    const double one_minus_p = 1.0 - p;
    const double up = p * risk_free_rate;
    const double down = one_minus_p * risk_free_rate;
    const int sign = option_type_sign(type);

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int OUTPUTS_PER_THREAD = h.OUTPUTS_PER_THREAD;

    const int thread_per_block = THREADS_PER_BLOCK;
    int num_blocks = std::ceil((n + 1) * 1.0 / (OUTPUTS_PER_THREAD * thread_per_block));

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 1) * sizeof(double));

    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / thread_per_block);
    FUNC_NAME(fill_st_buffer_kernel)<<<fill_num_blocks, thread_per_block>>>(st_buffer, S, K, u,
                                                                            sign, n);

    FUNC_NAME(compute_first_layer_kernel)<OUTPUTS_PER_THREAD>
        <<<num_blocks, thread_per_block>>>(d_option_values, st_buffer, n);
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / (thread_per_block * OUTPUTS_PER_THREAD));
        FUNC_NAME(compute_next_layer_kernel)<THREADS_PER_BLOCK, OUTPUTS_PER_THREAD>
            <<<num_blocks, thread_per_block>>>(d_option_values, d_option_values_next, st_buffer, up,
                                               down, level, n);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(st_buffer);
        checkCuda(cudaGetLastError());
        return h_s_store;
}

template double FUNC_NAME(
    vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_YUNROLL_VTILE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

#ifdef DO_CARTESIAN_PRODUCT
#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE(ID, A, B, C, D, E, Y) \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>(   \
        const double S, const double K, const double T, const double r, const double sigma,    \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE,
               HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif
