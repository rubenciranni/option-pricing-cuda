#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define IMPL_NAME stprcmp

#define THREADS_PER_BLOCK 1024

__global__ void FUNC_NAME(fill_st_buffer_kernel_batch)(double* __restrict__ d_st_buffer,
                                                       const double* d_S, const double* d_K,
                                                       const double* d_u, const int* d_sign,
                                                       const int* d_n, const int max_n) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (threadId > 2 * n) return;

    double S = d_S[option_idx];
    double K = d_K[option_idx];
    double u = d_u[option_idx];
    int sign = d_sign[option_idx];

    int local_index = ((threadId - n) & 1) * (n + 1) + (threadId) / 2;
    int idx = option_idx * (2 * max_n + 2) + local_index;
    d_st_buffer[idx] = max(sign * (S * pow(u, threadId - n) - K), 0.0);
}

__global__ void FUNC_NAME(compute_first_layer_kernel_batch)(double* d_option_values,
                                                            const double* d_st_buffer,
                                                            const int* d_n, const int max_n) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (threadId > n) return;
    int exp = 2 * threadId;
    int local_index = ((exp - n) & 1) * (n + 1) + exp / 2;
    int st_idx = option_idx * (2 * max_n + 2) + local_index;
    int out_idx = option_idx * (max_n + 1) + threadId;
    d_option_values[out_idx] = d_st_buffer[st_idx];
}

__global__ void FUNC_NAME(compute_next_layer_batch)(double* d_option_values,
                                                    double* d_option_values_next,
                                                    const double* d_st_buffer, const double* d_up,
                                                    const double* d_down, const int* d_n,
                                                    const int max_n, const int level) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];
    int out_base = option_idx * (max_n + 1);

    if (level >= n) {
        if (threadId <= n) {
            d_option_values_next[out_base + threadId] = d_option_values[out_base + threadId];
        }
        return;
    }

    if (threadId > level) return;

    double up = d_up[option_idx];
    double down = d_down[option_idx];

    double hold =
        up * d_option_values[out_base + threadId + 1] + down * d_option_values[out_base + threadId];
    int exp = 2 * threadId - level;
    int local_index = (exp & 1) * (n + 1) + (exp + n) / 2;
    int st_idx = option_idx * (2 * max_n + 2) + local_index;
    double exercise = d_st_buffer[st_idx];
    d_option_values_next[out_base + threadId] = max(hold, exercise);
}

double FUNC_NAME(vanilla_american_binomial_cuda)(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {
    std::vector<PricingInput> runs(1);
    runs[0].S = S;
    runs[0].K = K;
    runs[0].T = T;
    runs[0].r = r;
    runs[0].sigma = sigma;
    runs[0].q = q;
    runs[0].n = n;
    runs[0].type = type;

    std::vector<double> results(1);
    FUNC_NAME(vanilla_american_binomial_cuda_batch)(runs, results);
    return results[0];
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

    double *d_option_values, *d_option_values_next;
    size_t grid_size = num_runs * (max_n + 1);
    cudaMalloc(&d_option_values, grid_size * sizeof(double));
    cudaMalloc(&d_option_values_next, grid_size * sizeof(double));

    double* d_st_buffer;
    size_t st_size = num_runs * (2 * max_n + 2);
    cudaMalloc(&d_st_buffer, st_size * sizeof(double));

    const int num_threads = THREADS_PER_BLOCK;
    dim3 num_blocks(std::ceil((max_n + 1) * 1.0 / num_threads), num_runs);

    int fill_blocks_x = std::ceil((2 * max_n + 1) * 1.0 / num_threads);
    dim3 fill_blocks(fill_blocks_x, num_runs);

    FUNC_NAME(fill_st_buffer_kernel_batch)<<<fill_blocks, num_threads>>>(d_st_buffer, d_S, d_K, d_u,
                                                                         d_sign, d_n_arr, max_n);

    FUNC_NAME(compute_first_layer_kernel_batch)<<<num_blocks, num_threads>>>(
        d_option_values, d_st_buffer, d_n_arr, max_n);
    for (int level = max_n - 1; level >= 0; level--) {
        FUNC_NAME(compute_next_layer_batch)<<<num_blocks, num_threads>>>(
            d_option_values, d_option_values_next, d_st_buffer, d_up, d_down, d_n_arr, max_n,
            level);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();

    std::vector<double> h_results_store(grid_size);
    cudaMemcpy(h_results_store.data(), d_option_values, grid_size * sizeof(double),
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
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(d_st_buffer);
    checkCuda(cudaGetLastError());

    return;
}
