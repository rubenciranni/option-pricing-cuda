#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define IMPL_NAME bkdstprcmp

#define THREADS_PER_BLOCK 1024

__global__ void FUNC_NAME(fill_st_buffer_kernel)(double* __restrict__ st_buffer, const double S,
                                                 const double K, const double u, const int sign,
                                                 const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= 2 * n + 2) return;

    int i = threadId % (n + 1);
    st_buffer[threadId] = fmax(sign * fma(S, pow(u, (double)2 * i - n + (threadId > n)), -K), 0.0);
}

__global__ void FUNC_NAME(compute_next_layer_kernel)(double* layer_values_read,
                                                     double* layer_values_write, double* st_buffer,
                                                     const double up, const double down,
                                                     const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= level + 1) return;

    double hold = up * layer_values_read[threadId + 1] + down * layer_values_read[threadId];
    double exercise = st_buffer[threadId];
    layer_values_write[threadId] = max(hold, exercise);
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

    double* st_buffer_d;
    cudaMalloc(&st_buffer_d, (2 * n + 2) * sizeof(double));

    int num_blocks = std::ceil((2 * n + 2) * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(fill_st_buffer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(st_buffer_d, S, K, u, sign,
                                                                        n);

    // Layer n is the first n + 1 entries of st_buffer
    cudaMemcpy(layer_values_read_d, st_buffer_d, (n + 1) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // Layers n - 1 to 0
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        int st_buffer_offset = ((n - level) % 2) * (n + 1) + (n - level) / 2;
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_d + st_buffer_offset, up, down,
            level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }
    cudaDeviceSynchronize();

    double value_h;
    cudaMemcpy(&value_h, layer_values_read_d, (1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_d);

    checkCuda(cudaGetLastError());
    return value_h;
}
