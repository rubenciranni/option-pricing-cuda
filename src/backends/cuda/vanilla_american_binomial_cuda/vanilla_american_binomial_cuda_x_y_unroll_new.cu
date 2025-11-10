#include <assert.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

__global__ void fill_pricing_x_y_unroll_new(double* __restrict__ buffer, const double S,
                                            const double K, const double u, const int sign,
                                            const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > 2 * n) return;
    buffer[threadId] = fmax(sign * (S * pow(u, (double)threadId - n) - K), 0.0);
}

__global__ void first_layer_kernel_x_y_unroll_new(double* d_option_values,
                                                  double* __restrict__ st_buffer, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > n) return;
    int idx_uns = 2 * threadId;
    d_option_values[threadId] = st_buffer[idx_uns];
}

template <const int UNROLL_FACTOR, const int OUTPUTS_PER_THREAD, const int MAX_LEVEL_SIZE>
__global__ void vanilla_american_binomial_cuda_kernel_x_y_unroll_new(
    const double* __restrict__ d_option_values, double* d_option_values_next,
    const double* __restrict__ st_buffer, const double prob_up, const double prob_down,
    const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    threadId *= OUTPUTS_PER_THREAD;
    if (threadId - OUTPUTS_PER_THREAD + 1 > level) return;
    threadId = max(min(threadId, level - OUTPUTS_PER_THREAD + 1), 0);

    double res[MAX_LEVEL_SIZE + 1];
#pragma unroll
    for (int i = 0; i <= MAX_LEVEL_SIZE; i++) {
        res[i] = d_option_values[threadId + i];
    }
    double exponents[MAX_LEVEL_SIZE * 2];

    const int base_idx = (2 * (threadId) - (level + UNROLL_FACTOR - 1)) + n;
#pragma unroll
    for (int i = 0; i < MAX_LEVEL_SIZE * 2; i++) {
        exponents[i] = st_buffer[base_idx + i];
    }

#pragma unroll
    for (int delta_level = UNROLL_FACTOR - 1; delta_level >= 0; delta_level--) {
        for (int delta_id = 0; delta_id <= delta_level + OUTPUTS_PER_THREAD - 1; delta_id++) {
            double exponent = exponents[2 * delta_id - delta_level + UNROLL_FACTOR - 1];
            res[delta_id] =
                fmax(exponent, fma(prob_up, res[delta_id + 1], fma(prob_down, res[delta_id], 0.0)));
        }
    }
#pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        d_option_values_next[threadId + i] = res[i];
    }
}

__global__ void single_vanilla_american_binomial_cuda_kernel_x_y_unroll_new(
    double* d_option_values, double* d_option_values_next, double* st_buffer, const double prob_up,
    const double prob_down, const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > level) return;
    double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
    int exp = 2 * threadId - level;
    double exercise = st_buffer[exp + n];
    d_option_values_next[threadId] = max(hold, exercise);
}

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_x_y_unroll_new(const double S, const double K, const double T,
                                                     const double r, const double sigma,
                                                     const double q, const int n,
                                                     const OptionType type) {
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
    constexpr int thread_per_block = THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;
    constexpr int OUTPUTS_PER_THREAD = h.OUTPUTS_PER_THREAD;
    constexpr int MAX_LEVEL_SIZE = h.MAX_LEVEL_SIZE;

    int num_blocks = std::ceil((n + 1) * 1.0 / thread_per_block);

    double *d_option_values, *d_option_values_next;
    // int buffer_dim = std::ceil((n+1)*1.0 / OUTPUTS_PER_THREAD)*OUTPUTS_PER_THREAD;
    int buffer_dim = n + 1;
    cudaMalloc(&d_option_values, buffer_dim * sizeof(double));
    cudaMalloc(&d_option_values_next, buffer_dim * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 2) * sizeof(double));
    cudaDeviceSynchronize();
    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / 1024);
    fill_pricing_x_y_unroll_new<<<fill_num_blocks, 1024>>>(st_buffer, S, K, u, sign, n);

    first_layer_kernel_x_y_unroll_new<<<num_blocks, thread_per_block>>>(d_option_values, st_buffer,
                                                                        n);
    int level = n;
    int _iter = 0;
    bool profiling = false;
    cudaDeviceSynchronize();
    for (; level >= UNROLL_FACTOR && level > OUTPUTS_PER_THREAD; level -= UNROLL_FACTOR) {
        num_blocks =
            std::ceil((level - UNROLL_FACTOR + 1) * 1.0 / (thread_per_block * OUTPUTS_PER_THREAD));
        if ((_iter % 100) == 0 && profiling) {
            std::string kernel_label = "lvl_" + std::to_string(level);
            nvtxRangePushA(kernel_label.c_str());
            std::cout << "nvtx range " << kernel_label << "\n";
            cudaProfilerStart();
            vanilla_american_binomial_cuda_kernel_x_y_unroll_new<UNROLL_FACTOR, OUTPUTS_PER_THREAD,
                                                                 MAX_LEVEL_SIZE>
                <<<num_blocks, thread_per_block>>>(d_option_values, d_option_values_next, st_buffer,
                                                   up, down, level - UNROLL_FACTOR, n);
            cudaDeviceSynchronize();
            cudaProfilerStop();
            nvtxRangePop();
        } else {
            vanilla_american_binomial_cuda_kernel_x_y_unroll_new<UNROLL_FACTOR, OUTPUTS_PER_THREAD,
                                                                 MAX_LEVEL_SIZE>
                <<<num_blocks, thread_per_block>>>(d_option_values, d_option_values_next, st_buffer,
                                                   up, down, level - UNROLL_FACTOR, n);
        }
        std::swap(d_option_values, d_option_values_next);
        _iter++;
    }

    for (; level >= 1; level--) {
        num_blocks = std::ceil((level)*1.0 / thread_per_block);
        single_vanilla_american_binomial_cuda_kernel_x_y_unroll_new<<<num_blocks,
                                                                      thread_per_block>>>(
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

template double
vanilla_american_binomial_cuda_x_y_unroll_new<DEFAULT_HYPERPARAMS_CUDA_XY_UNROLL_NEW>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);
