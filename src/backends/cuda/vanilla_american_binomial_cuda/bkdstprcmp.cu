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

// Batch processing kernels
__global__ void FUNC_NAME(fill_st_buffer_kernel_batch)(double* __restrict__ d_st_buffer,
                                                       const double* d_S, const double* d_K,
                                                       const double* d_u, const int* d_sign,
                                                       const int* d_n, const int max_n) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (threadId >= 2 * n + 2) return;

    double S = d_S[option_idx];
    double K = d_K[option_idx];
    double u = d_u[option_idx];
    int sign = d_sign[option_idx];

    int i = threadId % (n + 1);
    int idx = option_idx * (2 * max_n + 2) + threadId;
    d_st_buffer[idx] = fmax(sign * fma(S, pow(u, (double)2 * i - n + (threadId > n)), -K), 0.0);
}

__global__ void FUNC_NAME(compute_next_layer_kernel_batch)(
    double* d_layer_values_read, double* d_layer_values_write, const double* d_st_buffer,
    const double* d_up, const double* d_down, const int* d_n, const int max_n, const int level) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (level >= n) {
        if (threadId <= n) {
            int idx = option_idx * (max_n + 1) + threadId;
            d_layer_values_write[idx] = d_layer_values_read[idx];
        }
        return;
    }

    if (threadId >= level + 1) return;

    double up = d_up[option_idx];
    double down = d_down[option_idx];

    int layer_base = option_idx * (max_n + 1);
    double hold = up * d_layer_values_read[layer_base + threadId + 1] +
                  down * d_layer_values_read[layer_base + threadId];

    int st_buffer_offset = ((n - level) % 2) * (n + 1) + (n - level) / 2;
    int st_idx = option_idx * (2 * max_n + 2) + st_buffer_offset + threadId;
    double exercise = d_st_buffer[st_idx];
    d_layer_values_write[layer_base + threadId] = max(hold, exercise);
}

void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    int max_n = 0;
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    std::vector<double> h_up(num_runs), h_down(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];
        if (run.n > max_n) max_n = run.n;

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;

        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_up[i] = p * risk_free_rate;
        h_down[i] = one_minus_p * risk_free_rate;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);
    }

    double *d_S, *d_K, *d_u, *d_up, *d_down;
    int *d_n_arr, *d_sign;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(double));
    cudaMalloc(&d_down, num_runs * sizeof(double));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    double *d_layer_values_read, *d_layer_values_write;
    size_t layer_size = num_runs * (max_n + 1);
    cudaMalloc(&d_layer_values_read, layer_size * sizeof(double));
    cudaMalloc(&d_layer_values_write, layer_size * sizeof(double));

    double* d_st_buffer;
    size_t st_size = num_runs * (2 * max_n + 2);
    cudaMalloc(&d_st_buffer, st_size * sizeof(double));

    const int num_threads = THREADS_PER_BLOCK;
    int fill_blocks_x = std::ceil((2 * max_n + 2) * 1.0 / num_threads);
    dim3 fill_blocks(fill_blocks_x, num_runs);

    FUNC_NAME(fill_st_buffer_kernel_batch)<<<fill_blocks, num_threads>>>(
        d_st_buffer, d_S, d_K, d_u, d_sign, d_n_arr, max_n);

    // Copy first layer (layer n) from st_buffer
    for (size_t i = 0; i < num_runs; ++i) {
        int n = h_n[i];
        cudaMemcpy(d_layer_values_read + i * (max_n + 1),
                   d_st_buffer + i * (2 * max_n + 2), (n + 1) * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    // Compute layers n - 1 to 0
    for (int level = max_n - 1; level >= 0; level--) {
        int num_blocks_x = std::ceil((level + 1) * 1.0 / num_threads);
        dim3 num_blocks(num_blocks_x, num_runs);
        FUNC_NAME(compute_next_layer_kernel_batch)<<<num_blocks, num_threads>>>(
            d_layer_values_read, d_layer_values_write, d_st_buffer, d_up, d_down, d_n_arr, max_n,
            level);
        std::swap(d_layer_values_read, d_layer_values_write);
    }
    cudaDeviceSynchronize();

    std::vector<double> h_results_store(layer_size);
    cudaMemcpy(h_results_store.data(), d_layer_values_read, layer_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_runs; ++i) {
        out[i] = h_results_store[i * (max_n + 1)];
    }

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_layer_values_read);
    cudaFree(d_layer_values_write);
    cudaFree(d_st_buffer);
    checkCuda(cudaGetLastError());
}
