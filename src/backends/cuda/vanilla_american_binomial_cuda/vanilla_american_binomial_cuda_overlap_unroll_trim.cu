#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

// TPB 128 UF 35 ~2.6ms on 10k
// TPB 256 UF 32 ~660ms on 250k

#define IMPL_NAME overlap_unroll_trimotm

#define THREADS_PER_BLOCK 128
#define UNROLL_FACTOR 37

__global__ void FUNC_NAME(fill_st_buffers_kernel)(double* __restrict__ st_buffer_bank0,
                                                  double* __restrict__ st_buffer_bank1,
                                                  const double S, const double K, const double u,
                                                  const int sign, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= n + 1) return;

    double u_pow_2_threadId = pow(u, (double)2 * threadId);
    double u_pow_minus_n = pow(u, (double)-n);

    // entry i stores value corresponding to exponent 2*i - n
    st_buffer_bank0[threadId] = fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n, -K), 0.0);

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[threadId] = fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n * u, -K), 0.0);
}

__global__ void FUNC_NAME(compute_next_layers_kernel)(
    double* __restrict__ layer_values_read, double* __restrict__ layer_values_write,
    double* __restrict__ st_buffer_bank0, double* __restrict__ st_buffer_bank1, const double up,
    const double down, const int level, const int n, const int last) {
    __shared__ double layer_values_tile[2][THREADS_PER_BLOCK + 1];

    int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    int tile_base = tile_stride * blockIdx.x;
    int node_id = tile_base + threadIdx.x;

    if (node_id >= n + 1) return;

    layer_values_tile[0][threadIdx.x] = layer_values_read[node_id];

    __syncthreads();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int read_idx = i % 2;
        int write_idx = (i + 1) % 2;

        int current_level = level + UNROLL_FACTOR - 1 - i;
        double* st_buffer_bank = (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        double hold = fma(up, layer_values_tile[read_idx][threadIdx.x + 1],
                          down * layer_values_tile[read_idx][threadIdx.x]);
        int st_index = node_id + (n - current_level) / 2;
        double exercise = st_buffer_bank[st_index];
        layer_values_tile[write_idx][threadIdx.x] = fmax(hold, exercise);

        __syncthreads();
    }

    if (node_id <= n && threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[node_id] = layer_values_tile[UNROLL_FACTOR % 2][threadIdx.x];
    }
}

/*
    At each layer l exercise value of node i (from the bottom) is calculated with the following
    exponent: 2*i - l = 2 * (i + (n - l) / 2) - n           if (n - l) even
    the correspoding value is stored at st_buffer_bank0[i + (n - l) / 2]

    2*i - l = 2 * (i + (n - l - 1) / 2) - n + 1   if (n - l) odd
    the correspoding value is stored at st_buffer_bank1[i + (n - l) / 2]
*/
__global__ void FUNC_NAME(compute_next_layer_kernel)(double* __restrict__ layer_values_read,
                                                     double* __restrict__ layer_values_write,
                                                     double* __restrict__ st_buffer_bank0,
                                                     double* __restrict__ st_buffer_bank1,
                                                     const double up, const double down,
                                                     const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= level + 1) return;

    double hold = up * layer_values_read[threadId + 1] + down * layer_values_read[threadId];
    double* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    double exercise = st_buffer_bank[threadId + (n - level) / 2];
    layer_values_write[threadId] = fmax(hold, exercise);
}

int FUNC_NAME(search_bound)(const int n, const double S, const double K, const double u,
                            const int sign) {
    if (sign == 1) return n;

    int lower = 0;
    int upper = n;
    while (lower < upper) {
        int mid = lower + (upper - lower + 1) / 2;
        double S_mid_n = sign * (S * std::pow(u, mid * 2 - n) - K);
        if (S_mid_n < 0.)
            upper = mid - 1;
        else
            lower = mid;
    }
    return lower;
}

double FUNC_NAME(vanilla_american_binomial_cuda)(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {
    const double delta_t = T / n;
    const double u = std::exp(sigma * std::sqrt(delta_t));
    const double d = 1.0 / u;
    const double p = (exp((r - q) * delta_t) - d) / (u - d);
    const double discount = std::exp(-r * delta_t);
    const double up = p * discount;
    const double down = (1.0 - p) * discount;
    const int sign = option_type_sign(type);

    double *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, (n + 1) * sizeof(double));
    cudaMalloc(&layer_values_write_d, (n + 1) * sizeof(double));

    double *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, (n + 1) * sizeof(double));
    cudaMalloc(&st_buffer_bank1_d, (n + 1) * sizeof(double));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(fill_st_buffers_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
        st_buffer_bank0_d, st_buffer_bank1_d, S, K, u, sign, n);

    // Layer n is the first n + 1 entries of st_buffer_bank0_d
    cudaMemcpy(layer_values_read_d, st_buffer_bank0_d, (n + 1) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    int bound = FUNC_NAME(search_bound)(n, S, K, u, sign);
    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        int num_nodes = std::min(level, bound);
        num_blocks =
            std::ceil((num_nodes + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        FUNC_NAME(compute_next_layers_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up,
            down, level, n, num_nodes);
        std::swap(layer_values_read_d, layer_values_write_d);
    }
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up,
            down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    cudaDeviceSynchronize();

    double value_h;
    cudaMemcpy(&value_h, layer_values_read_d, (1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);

    return value_h;
}
