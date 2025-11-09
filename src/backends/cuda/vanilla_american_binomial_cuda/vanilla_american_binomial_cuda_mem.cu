#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "assert.h"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

// TpB 128 UF 64 OpT 64 ~2.8ms on 10k

#define THREADS_PER_BLOCK 128
#define UNROLL_FACTOR 64
#define OUTPUTS_PER_BLOCK 64

__global__ void fill_st_buffer_kernel(double* __restrict__ st_buffer, const double S,
                                      const double K, const double u, const int sign, const int n,
                                      const int parity) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= n + 1) return;

    st_buffer[threadId] = fmax(sign * fma(S, pow(u, (double)(2 * threadId - n + parity)), -K), 0.0);
}

__global__ void compute_next_layers_kernel(double* layer_values_read, double* layer_values_write,
                                           double* st_buffer_bank0, double* st_buffer_bank1,
                                           const double up, const double down, int level, int n) {
    const int block_write_off = OUTPUTS_PER_BLOCK;
    int threadId = blockIdx.x * block_write_off + threadIdx.x;

    constexpr int TILE_SIZE = OUTPUTS_PER_BLOCK + UNROLL_FACTOR;
    __shared__ double layer_values_tile_read_array[TILE_SIZE];
    __shared__ double layer_values_tile_write_array[TILE_SIZE];

    double* layer_values_tile_read = layer_values_tile_read_array;
    double* layer_values_tile_write = layer_values_tile_write_array;

    const int block_offset = blockIdx.x * block_write_off;
    for (int i = threadIdx.x; i < TILE_SIZE; i += THREADS_PER_BLOCK) {
        if (block_offset + i <= level + UNROLL_FACTOR)
            layer_values_tile_read[i] = layer_values_read[block_offset + i];
    }
    __syncthreads();

    int tile_size = TILE_SIZE - 1;
#pragma unroll
    for (int i = UNROLL_FACTOR - 1; i >= 0; i--) {
        const int current_level = level + i;
        for (int offset = threadIdx.x; offset < tile_size; offset += THREADS_PER_BLOCK) {
            int globalIdx = block_offset + offset;
            if (globalIdx <= current_level) {
                double* st_buffer_bank =
                    (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
                double exercise = st_buffer_bank[globalIdx + (n - current_level) / 2];
                double hold = fmax(exercise, fma(up, layer_values_tile_read[offset + 1],
                                                 fma(down, layer_values_tile_read[offset], 0.0)));
                layer_values_tile_write[offset] = fmax(hold, exercise);
            }
        }

        __syncthreads();

        double* tmp = layer_values_tile_read;
        layer_values_tile_read = layer_values_tile_write;
        layer_values_tile_write = tmp;

        tile_size--;
    }

    if (threadId <= level && threadIdx.x < OUTPUTS_PER_BLOCK) {
        layer_values_write[threadId] = layer_values_tile_read[threadIdx.x];
    }
}

__global__ void compute_next_layer_kernel(double* layer_values_read, double* layer_values_write,
                                          double* st_buffer_bank0, double* st_buffer_bank1,
                                          const double up, const double down, const int level,
                                          const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= level + 1) return;

    /*
    At each layer l exercise value of node i (from the bottom) is calculated with the following
    exponent: 2*i - l = 2 * (i + (n - l) / 2) - n           if (n - l) even the correspoding value
    is stored at st_buffer_bank0[i + (n - l) / 2]

    2*i - l = 2 * (i + (n - l - 1) / 2) - n + 1   if (n - l) odd
    the correspoding value is stored at st_buffer_bank1[i + (n - l) / 2]
    */

    double hold = up * layer_values_read[threadId + 1] + down * layer_values_read[threadId];
    double* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    double exercise = st_buffer_bank[threadId + (n - level) / 2];
    layer_values_write[threadId] = fmax(hold, exercise);
}

double vanilla_american_binomial_cuda_x_y_unroll_tile_banked(const double S, const double K,
                                                             const double T, const double r,
                                                             const double sigma, const double q,
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
    fill_st_buffer_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(st_buffer_bank0_d, S, K, u, sign, n,
                                                             0);
    fill_st_buffer_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(st_buffer_bank1_d, S, K, u, sign, n,
                                                             1);

    // Layer n is the first n + 1 entries of st_buffer
    cudaMemcpy(layer_values_read_d, st_buffer_bank0_d, (n + 1) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    int level = n;
    for (; level >= UNROLL_FACTOR && level > 1; level -= UNROLL_FACTOR) {
        num_blocks = std::ceil((level - UNROLL_FACTOR + 1) * 1.0 / OUTPUTS_PER_BLOCK);
        compute_next_layers_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up,
            down, level - UNROLL_FACTOR, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    for (; level >= 1; level -= 1) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        compute_next_layer_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up,
            down, level - 1, n);
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
