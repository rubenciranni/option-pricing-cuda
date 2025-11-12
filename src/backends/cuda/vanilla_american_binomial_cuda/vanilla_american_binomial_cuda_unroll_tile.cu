#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

#define IMPL_NAME unroll_tile

#define INDEX_BUFFER(idx, buffer_size) idx

__global__ void FUNC_NAME(fill_st_buffer_kernel)(double* __restrict__ buffer, const double S,
                                                 const double K, const double u, const int sign,
                                                 const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > 2 * n) return;
    buffer[INDEX_BUFFER(threadId, n)] = fmax(sign * (S * pow(u, threadId - n) - K), 0.0);
}

__global__ void FUNC_NAME(compute_first_layer_kernel)(double* d_option_values,
                                                      double* __restrict__ st_buffer, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > n) return;
    d_option_values[threadId] = st_buffer[INDEX_BUFFER(2 * threadId, n)];
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__device__ __forceinline__ double FUNC_NAME(compute_index)(
    const double* past_values, const double* st_buffer, const double prob_up,
    const double prob_down, const int idx_tile, const int idx, const int level, const int n) {
    double res[UNROLL_FACTOR + 1];
#pragma unroll
    for (int i = 0; i <= UNROLL_FACTOR; i++) {
        res[i] = past_values[idx_tile + i];
    }
#pragma unroll
    for (int delta_level = UNROLL_FACTOR - 1; delta_level >= 0; delta_level--) {
#pragma unroll
        for (int delta_id = 0; delta_id <= delta_level; delta_id++) {
            int exponent = 2 * (delta_id + idx) - delta_level + UNROLL_FACTOR - 1;
            res[delta_id] =
                fmax(st_buffer[INDEX_BUFFER(exponent, THREADS_PER_BLOCK + UNROLL_FACTOR)],
                     fma(prob_up, res[delta_id + 1], prob_down * res[delta_id]));
        }
    }
    return res[0];
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel)(const double* d_option_values,
                                                      double* d_option_values_next,
                                                      const double* __restrict__ st_buffer,
                                                      const double prob_up, const double prob_down,
                                                      const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double tile[UNROLL_FACTOR + THREADS_PER_BLOCK];
    __shared__ double pow_tile[2 * (THREADS_PER_BLOCK + UNROLL_FACTOR) + 1];

#pragma unroll
    for (int i = threadIdx.x; i < UNROLL_FACTOR + THREADS_PER_BLOCK; i += THREADS_PER_BLOCK) {
        if (blockIdx.x * blockDim.x + i <= level + UNROLL_FACTOR) {
            tile[i] = d_option_values[blockIdx.x * blockDim.x + i];
        }
    }

#pragma unroll
    for (int i = threadIdx.x; i <= 2 * (UNROLL_FACTOR + THREADS_PER_BLOCK);
         i += THREADS_PER_BLOCK) {
        pow_tile[INDEX_BUFFER(i, THREADS_PER_BLOCK + UNROLL_FACTOR)] = st_buffer[INDEX_BUFFER(
            2 * (blockIdx.x * blockDim.x) + i - UNROLL_FACTOR + 1 - level + n, n)];
    }

    __syncthreads();

    if (threadId > level) return;

    double res = FUNC_NAME(compute_index)<THREADS_PER_BLOCK, UNROLL_FACTOR>(
        tile, pow_tile, prob_up, prob_down, threadIdx.x, threadIdx.x, level, n);
    d_option_values_next[threadId] = res;
}

__global__ void FUNC_NAME(compute_next_layer_kernel)(double* d_option_values,
                                                     double* d_option_values_next,
                                                     double* st_buffer, const double prob_up,
                                                     const double prob_down, const int level,
                                                     const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId > level) return;
    double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
    double exercise = st_buffer[2 * threadId - level + n];
    d_option_values_next[threadId] = max(hold, exercise);
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
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    const int thread_per_block = THREADS_PER_BLOCK;
    int num_blocks = std::ceil((n + 1) * 1.0 / thread_per_block);
    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 1) * sizeof(double));

    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / 1024);
    FUNC_NAME(fill_st_buffer_kernel)<<<fill_num_blocks, 1024>>>(st_buffer, S, K, u, sign, n);

    FUNC_NAME(compute_first_layer_kernel)<<<num_blocks, thread_per_block>>>(d_option_values,
                                                                            st_buffer, n);
    int level = n;
    for (; level >= UNROLL_FACTOR; level -= UNROLL_FACTOR) {
        num_blocks = std::ceil((level - UNROLL_FACTOR + 1) * 1.0 / thread_per_block);
        FUNC_NAME(compute_next_layers_kernel)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks, thread_per_block>>>(d_option_values, d_option_values_next, st_buffer, up,
                                               down, level - UNROLL_FACTOR, n);
        std::swap(d_option_values, d_option_values_next);
    }

    for (; level >= 1; level--) {
        num_blocks = std::ceil((level)*1.0 / thread_per_block);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, thread_per_block>>>(
            d_option_values, d_option_values_next, st_buffer, up, down, level - 1, n);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();

    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(st_buffer);
    return h_s_store;
}

template double FUNC_NAME(vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_UNROLL_TILE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

#ifdef DO_CARTESIAN_PRODUCT
#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_UNROLL_TILE

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_UNROLL_TILE(ID, A, B, C, D, E, Y)         \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>( \
        const double S, const double K, const double T, const double r, const double sigma,  \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_UNROLL_TILE, HYPERPARAMS_CART_PRODUCT,
               NULL)

#endif
#endif
