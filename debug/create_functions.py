from pathlib import Path
signatures =[]
map_include = []
for thread_per_block in [16,64,128]:
    for unroll_factor in [1,2,16]:
        for thread_output in [1,2,4,16]:
            file=f"""
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define THREADS_PER_BLOCK {thread_per_block}
#define UNROLL_FACTOR {unroll_factor}
#define OUTPUTS_PER_THREAD {thread_output}
#define MAX_LEVEL_SIZE (UNROLL_FACTOR + OUTPUTS_PER_THREAD - 1)

__global__ void fill_pricing_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(double* __restrict__ buffer, const double S, const double K,
                                        const double u, const int sign, const int n) {{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > 2 * n) return;
    buffer[((threadId - n) & 1) * (n + 1) + (threadId) / 2] =
        max(sign * (S * pow(u, threadId - n) - K), 0.0);
}}

__global__ void first_layer_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(double* d_option_values,
                                              double* __restrict__ st_buffer, const int n) {{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > n) return;
    int idx_uns = 2 * threadId;
    d_option_values[threadId] = st_buffer[((idx_uns - n) & 1) * (n + 1) + idx_uns / 2];
}}


__global__ void vanilla_american_binomial_cuda_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(
    const double* d_option_values, double* d_option_values_next,
    const double* __restrict__ st_buffer, const double prob_up, const double prob_down,
    const int level, const int n) {{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    threadId *= OUTPUTS_PER_THREAD;
    if (threadId - OUTPUTS_PER_THREAD + 1 > level) return;
    threadId = min(threadId, level - OUTPUTS_PER_THREAD + 1);

    const int last_layer_size = UNROLL_FACTOR + OUTPUTS_PER_THREAD - 1;
    double res[ MAX_LEVEL_SIZE+1 ];
#pragma unroll
    for (int i = 0; i <= MAX_LEVEL_SIZE; i++) {{
        res[i] = d_option_values[threadId + i];
    }}
#pragma unroll
    for (int delta_level = UNROLL_FACTOR - 1; delta_level >= 0; delta_level--) {{
        for (int delta_id = 0; delta_id <= delta_level + OUTPUTS_PER_THREAD - 1; delta_id++) {{
            int exponent = 2 * (threadId + delta_id) - level - delta_level + n;
            int buf_idx = ((exponent - n) & 1) * (n + 1) + (exponent / 2);

            res[delta_id] =
                fmax(st_buffer[buf_idx], prob_up * res[delta_id + 1] + prob_down * res[delta_id]);
        }}
    }}

    for(int i = 0; i < OUTPUTS_PER_THREAD; i++) {{
        d_option_values_next[threadId + i] = res[i];
    }}
}}

__global__ void single_vanilla_american_binomial_cuda_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(
    double* d_option_values, double* d_option_values_next, double* st_buffer, const double prob_up,
    const double prob_down, const int level, const int n) {{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > level) return;
    double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
    int exp = 2 * threadId - level;
    double exercise = st_buffer[(exp & 1) * (n + 1) + (exp + n) / 2];
    d_option_values_next[threadId] = max(hold, exercise);
}}

double vanilla_american_binomial_cuda_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {{
    const double deltaT = T / n;
    const double u = std::exp(sigma * std::sqrt(deltaT));
    const double d = 1.0 / u;
    const double p = (exp((r - q) * deltaT) - d) / (u - d);
    const double risk_free_rate = std::exp(-r * deltaT);
    const double one_minus_p = 1.0 - p;
    const double up = p * risk_free_rate;
    const double down = one_minus_p * risk_free_rate;
    const int sign = option_type_sign(type);

    const int thread_per_block = THREADS_PER_BLOCK;
    int num_blocks = std::ceil((n + 1) * 1.0 / thread_per_block);

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 2) * sizeof(double));

    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / 1024);
    fill_pricing_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}<<<fill_num_blocks, 1024>>>(st_buffer, S, K, u, sign, n);

    first_layer_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}<<<num_blocks, thread_per_block>>>(d_option_values, st_buffer, n);
    int level = n;
    for (; level >= UNROLL_FACTOR && level > OUTPUTS_PER_THREAD; level -= UNROLL_FACTOR) {{
        num_blocks = std::ceil((level - UNROLL_FACTOR + 1) * 1.0 / (thread_per_block*OUTPUTS_PER_THREAD));
        vanilla_american_binomial_cuda_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}<<<num_blocks, thread_per_block>>>(
            d_option_values, d_option_values_next, st_buffer, up, down, level - UNROLL_FACTOR, n);
        std::swap(d_option_values, d_option_values_next);
    }}

    for (; level >= 1; level--) {{
        num_blocks = std::ceil((level) * 1.0 / thread_per_block);
        single_vanilla_american_binomial_cuda_kernel_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}<<<num_blocks, thread_per_block>>>(
            d_option_values, d_option_values_next, st_buffer, up, down, level - 1, n);
        std::swap(d_option_values, d_option_values_next);
    }}
    cudaDeviceSynchronize();

    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(st_buffer);

    return h_s_store;
}}
"""
            name  = f"vanilla_american_binomial_cuda_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}"
            Path(f"debug/cuda/{name}.cu").write_text(file)
            signatures.append(f"double vanilla_american_binomial_cuda_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type);")
            map_include.append(f"{{\"double vanilla_american_binomial_cuda_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}\",double vanilla_american_binomial_cuda_x_y_unroll_{thread_per_block}_{unroll_factor}_{thread_output}}},")
print("\n".join(signatures))
print("\n".join(map_include))
