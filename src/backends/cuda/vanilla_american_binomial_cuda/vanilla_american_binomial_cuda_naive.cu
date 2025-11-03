#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

__global__ void first_layer_kernel(double* d_option_values, int level, double S, double u, double K,
                                   const int sign) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id > level) return;
    double ST = S * pow(u, 2 * thread_id - level);
    d_option_values[thread_id] = max(0.0, sign * (ST - K));
}

__global__ void vanilla_american_binomial_cuda_kernel(double* d_option_values,
                                                      double* d_option_values_next, const double S,
                                                      const double K, const double up,
                                                      const double down, const double u,
                                                      const int level, const int sign) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id > level) return;
    double ST = S * pow(u, 2.0 * thread_id - level);
    double hold = up * d_option_values[thread_id + 1] + down * d_option_values[thread_id];
    double exercise = max(sign * (ST - K), 0.0);
    d_option_values_next[thread_id] = max(hold, exercise);
}

double vanilla_american_binomial_cuda_naive(const double S, const double K, const double T,
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

    const int num_threads = 1024;
    int num_blocks = std::ceil((n + 1) * 1.0 / num_threads);

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));

    first_layer_kernel<<<num_blocks, num_threads>>>(d_option_values, n, S, u, K, sign);
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
        cudaDeviceSynchronize();
        vanilla_american_binomial_cuda_kernel<<<num_blocks, num_threads>>>(
            d_option_values, d_option_values_next, S, K, up, down, u, level, sign);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    return h_s_store;
}

double vanilla_american_binomial_cuda_no_sync(const double S, const double K, const double T,
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

    const int num_threads = 1024;
    int num_blocks = std::ceil((n + 1) * 1.0 / num_threads);

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));

    first_layer_kernel<<<num_blocks, num_threads>>>(d_option_values, n, S, u, K, sign);
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
        vanilla_american_binomial_cuda_kernel<<<num_blocks, num_threads>>>(
            d_option_values, d_option_values_next, S, K, up, down, u, level, sign);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    return h_s_store;
}
