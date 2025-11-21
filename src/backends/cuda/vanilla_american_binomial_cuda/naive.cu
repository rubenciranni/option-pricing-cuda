#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define IMPL_NAME naive

__global__ void FUNC_NAME(compute_first_layer_kernel)(double* d_option_values, int level, double S,
                                                      double u, double K, const int sign) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id > level) return;
    double ST = S * pow(u, 2 * thread_id - level);
    d_option_values[thread_id] = max(0.0, sign * (ST - K));
}

__global__ void FUNC_NAME(compute_next_layer_kernel)(double* d_option_values,
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

    const int num_threads = 1024;
    int num_blocks = std::ceil((n + 1) * 1.0 / num_threads);

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));

    FUNC_NAME(compute_first_layer_kernel)<<<num_blocks, num_threads>>>(d_option_values, n, S, u, K,
                                                                       sign);
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, num_threads>>>(
            d_option_values, d_option_values_next, S, K, up, down, u, level, sign);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    checkCuda(cudaGetLastError());
    return h_s_store;
}

std::vector<double> FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput> runs) {
    // Create one stream per run
    std::vector<cudaStream_t> streams(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int num_threads = 1024;
    std::vector<double*> d_option_values_ds;
    std::vector<double*> d_option_values_next_ds;

    // Launch per-run work on its own stream using async alloc/copies/kernels
    for (size_t i = 0; i < runs.size(); ++i) {
        const PricingInput &run = runs[i];
        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;
        const double up = p * risk_free_rate;
        const double down = one_minus_p * risk_free_rate;
        const int sign = option_type_sign(run.type);

        double *d_option_values = nullptr;
        double *d_option_values_next = nullptr;
        cudaMallocAsync(&d_option_values, (run.n + 1) * sizeof(double), streams[i]);
        cudaMallocAsync(&d_option_values_next, (run.n + 1) * sizeof(double), streams[i]);

        int num_blocks = std::ceil((run.n + 1) * 1.0 / num_threads);
        FUNC_NAME(compute_first_layer_kernel)<<<num_blocks, num_threads, 0, streams[i]>>>(
            d_option_values, run.n, run.S, u, run.K, sign);
        d_option_values_ds.push_back(d_option_values);
        d_option_values_next_ds.push_back(d_option_values_next);
    }

    for (size_t i = 0; i < runs.size(); ++i) {
        const PricingInput &run = runs[i];
        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;
        const double up = p * risk_free_rate;
        const double down = one_minus_p * risk_free_rate;
        const int sign = option_type_sign(run.type);
        double *d_option_values = d_option_values_ds[i];
        double *d_option_values_next = d_option_values_next_ds[i];
        
        for (int level = run.n - 1; level >= 0; level--) {
            int num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
            FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, num_threads, 0, streams[i]>>>(
                d_option_values, d_option_values_next, run.S, run.K, up, down, u, level, sign);
            std::swap(d_option_values, d_option_values_next);
        }

    }

    // Collect results asynchronously and free per-run allocations on the same stream
    std::vector<double> results(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        cudaMemcpyAsync(&results[i], d_option_values_ds[i], sizeof(double), cudaMemcpyDeviceToHost,
                        streams[i]);
        cudaFreeAsync(d_option_values_ds[i], streams[i]);
        cudaFreeAsync(d_option_values_next_ds[i], streams[i]);
    }

    // Wait for all work to complete
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());

    // Destroy streams
    for (size_t i = 0; i < streams.size(); ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return results;
}
