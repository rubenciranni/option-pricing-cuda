#include <assert.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

/*
 * TpB 128, UF 128, OpT 1 <3.6 ms on 10K
 */

#define THREADS_PER_BLOCK 128
#define UNROLL_FACTOR 128
#define OUTPUTS_PER_THREAD 1
#define MAX_LEVEL_SIZE (UNROLL_FACTOR + OUTPUTS_PER_THREAD - 1)
#define CEIL_DIV(A, B) (((A) + (B)-1) / (B))

#define IMPL_NAME x_y_unroll_tile
#define CONCAT_IMPL(a, b) a##_##b
#define EXPAND_AND_CONCAT(a, b) CONCAT_IMPL(a, b)
#define KERNEL_NAME(func) EXPAND_AND_CONCAT(func, IMPL_NAME)

__global__ void KERNEL_NAME(fill_pricing)(double* __restrict__ buffer, const double S,
                                          const double K, const double u, const int sign,
                                          const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > 2 * n) return;
    buffer[threadId] = fmax(sign * fma(S, pow(u, (double)threadId - n), -K), 0.0);
}

__global__ void KERNEL_NAME(first_layer)(double* d_option_values, double* __restrict__ st_buffer,
                                         const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > n) return;
    int idx_uns = 2 * threadId;
    d_option_values[threadId] = st_buffer[idx_uns];
}

__global__ void KERNEL_NAME(vanilla_american_binomial_cuda_kernel)(
    const double* __restrict__ d_option_values, double* d_option_values_next,
    const double* __restrict__ st_buffer, const double prob_up, const double prob_down,
    const int level, const int n) {
    constexpr int values_tile_size = OUTPUTS_PER_THREAD * THREADS_PER_BLOCK + UNROLL_FACTOR;
    __shared__ double values_tile_read_array[values_tile_size];
    __shared__ double values_tile_write_array[values_tile_size];

    double* values_tile_write = values_tile_write_array;
    double* values_tile_read = values_tile_read_array;

    constexpr int prices_tile_size =
        2 * (OUTPUTS_PER_THREAD * THREADS_PER_BLOCK + UNROLL_FACTOR - 1);
    __shared__ double prices_tile[2][prices_tile_size / 2];

    const int value_bl_off = OUTPUTS_PER_THREAD * blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < values_tile_size; i += THREADS_PER_BLOCK) {
        if (value_bl_off + i <= level + UNROLL_FACTOR)
            values_tile_read[i] = d_option_values[value_bl_off + i];
    }

    // when threadIdx.x == 0 essentially
    const int min_exp_blk =
        2 * OUTPUTS_PER_THREAD * blockIdx.x * blockDim.x - UNROLL_FACTOR + 1 - level;

    // add + n offset to access the prices array (otherwise it's negative)
    const int prices_blk_off = min_exp_blk + n;
    for (int i = threadIdx.x; i < prices_tile_size; i += THREADS_PER_BLOCK) {
        if (i + prices_blk_off <= 2 * n) prices_tile[i % 2][i / 2] = st_buffer[prices_blk_off + i];
    }

    __syncthreads();

#pragma unroll
    for (int delta_level = UNROLL_FACTOR - 1; delta_level >= 0; delta_level--) {
        const int active_work_size = values_tile_size - UNROLL_FACTOR + delta_level;
#pragma unroll
        for (int v_tile_idx = threadIdx.x; v_tile_idx < active_work_size;
             v_tile_idx += THREADS_PER_BLOCK) {
            int p_tile_idx = 2 * v_tile_idx - delta_level + UNROLL_FACTOR - 1;
            double exercise = prices_tile[p_tile_idx % 2][p_tile_idx / 2];
            values_tile_write[v_tile_idx] =
                fmax(exercise, fma(prob_up, values_tile_read[v_tile_idx + 1],
                                   fma(prob_down, values_tile_read[v_tile_idx], 0.0)));
        }

        __syncthreads();

        double* tmp = values_tile_read;
        values_tile_read = values_tile_write;
        if (delta_level == 1)
            values_tile_write = d_option_values_next + OUTPUTS_PER_THREAD * blockIdx.x * blockDim.x;
        else
            values_tile_write = tmp;
    }
}

__global__ void KERNEL_NAME(single_vanilla_american_binomial_cuda_kernel)(
    double* d_option_values, double* d_option_values_next, double* st_buffer, const double prob_up,
    const double prob_down, const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > level) return;
    double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
    int exp = 2 * threadId - level;
    double exercise = st_buffer[exp + n];
    d_option_values_next[threadId] = max(hold, exercise);
}

double vanilla_american_binomial_cuda_x_y_unroll_tile(const double S, const double K,
                                                      const double T, const double r,
                                                      const double sigma, const double q,
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

    constexpr int thread_per_block = THREADS_PER_BLOCK;
    int num_blocks = std::ceil((n + 1) * 1.0 / thread_per_block);

    double *d_option_values, *d_option_values_next;
    // int buffer_dim = std::ceil((n+1)*1.0 / OUTPUTS_PER_THREAD)*OUTPUTS_PER_THREAD;
    int buffer_dim = (n + 1);
    cudaMalloc(&d_option_values, buffer_dim * sizeof(double));
    cudaMalloc(&d_option_values_next, buffer_dim * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 2) * sizeof(double));

    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / 1024);
    KERNEL_NAME(fill_pricing)<<<fill_num_blocks, 1024>>>(st_buffer, S, K, u, sign, n);

    KERNEL_NAME(first_layer)<<<num_blocks, thread_per_block>>>(d_option_values, st_buffer, n);
    int level = n;
#ifdef PROFILING
    int _iter = 0;
#endif

    for (; level >= UNROLL_FACTOR && level > OUTPUTS_PER_THREAD; level -= UNROLL_FACTOR) {
        num_blocks =
            std::ceil((level - UNROLL_FACTOR + 1) * 1.0 / (thread_per_block * OUTPUTS_PER_THREAD));
#ifdef PROFILING
        if ((_iter % 100) == 0) {
            std::string kernel_label = "lvl_" + std::to_string(level);
            nvtxRangePushA(kernel_label.c_str());
            cudaProfilerStart();
            KERNEL_NAME(vanilla_american_binomial_cuda_kernel)<<<num_blocks, thread_per_block>>>(
                d_option_values, d_option_values_next, st_buffer, up, down, level - UNROLL_FACTOR,
                n);
            cudaDeviceSynchronize();
            cudaProfilerStop();
            nvtxRangePop();
        } else {
            KERNEL_NAME(vanilla_american_binomial_cuda_kernel)<<<num_blocks, thread_per_block>>>(
                d_option_values, d_option_values_next, st_buffer, up, down, level - UNROLL_FACTOR,
                n);
        }
        _iter++;
#else
        KERNEL_NAME(vanilla_american_binomial_cuda_kernel)<<<num_blocks, thread_per_block>>>(
            d_option_values, d_option_values_next, st_buffer, up, down, level - UNROLL_FACTOR, n);
#endif
        std::swap(d_option_values, d_option_values_next);
    }

    for (; level >= 1; level--) {
        num_blocks = std::ceil((level)*1.0 / thread_per_block);
        KERNEL_NAME(single_vanilla_american_binomial_cuda_kernel)<<<num_blocks, thread_per_block>>>(
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
