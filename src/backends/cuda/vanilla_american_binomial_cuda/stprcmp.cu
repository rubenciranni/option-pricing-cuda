#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#include <vector>

#define IMPL_NAME stprcmp

#define THREADS_PER_BLOCK 1024

__global__ void FUNC_NAME(fill_st_buffer_kernel)(double* __restrict__ buffer, const double S,
                                                 const double K, const double u, const int sign,
                                                 const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    threadId = (threadId % (2 * n + 1));
    // st_buffer[0] = pow(u, -n)
    //  buffer[threadId] = max(sign*(S * pow(u, threadId-n) - K), 0.0);
    buffer[((threadId - n) & 1) * (n + 1) + (threadId) / 2] =
        max(sign * (S * pow(u, threadId - n) - K), 0.0);
}

__global__ void FUNC_NAME(compute_first_layer_kernel)(double* d_option_values, double* st_buffer,
                                                      const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId > n) return;
    int exp = 2 * threadId;
    // d_option_values[threadId] = st_buffer[2*threadId+1];
    d_option_values[threadId] = st_buffer[((exp - n) & 1) * (n + 1) + exp / 2];
}

__global__ void FUNC_NAME(compute_next_layer)(double* d_option_values, double* d_option_values_next,
                                              double* st_buffer, const double prob_up,
                                              const double prob_down, const int level,
                                              const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > level) return;

    //   double ST = S * pow(u, 2.0 * thread_id - level);
    double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
    // double exercise = st_buffer[2*threadId - level+n];
    int exp = 2 * threadId - level;
    double exercise = st_buffer[(exp & 1) * (n + 1) + (exp + n) / 2];
    d_option_values_next[threadId] = max(hold, exercise);
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

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
    double* st_buffer;
    cudaMalloc(&st_buffer, (2 * n + 2) * sizeof(double));

    int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(fill_st_buffer_kernel)<<<fill_num_blocks, THREADS_PER_BLOCK>>>(st_buffer, S, K, u,
                                                                             sign, n);

    FUNC_NAME(compute_first_layer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(d_option_values,
                                                                             st_buffer, n);
    for (int level = n - 1; level >= 0; level--) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        FUNC_NAME(compute_next_layer)<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_option_values, d_option_values_next, st_buffer, up, down, level, n);
        std::swap(d_option_values, d_option_values_next);
    }
    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(st_buffer);
    checkCuda(cudaGetLastError());
    return h_s_store;
}

std::vector<double> FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput> runs) {
    std::vector<cudaStream_t> streams(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) cudaStreamCreate(&streams[i]);

    std::vector<double*> d_option_values_ds;
    std::vector<double*> d_option_values_next_ds;
    std::vector<double*> st_buffer_ds;
    d_option_values_ds.reserve(runs.size());
    d_option_values_next_ds.reserve(runs.size());
    st_buffer_ds.reserve(runs.size());

    const int num_threads = THREADS_PER_BLOCK;

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

        double *d_option_values = nullptr;
        double *d_option_values_next = nullptr;
        double *st_buffer = nullptr;
        cudaMallocAsync(&d_option_values, (run.n + 1) * sizeof(double), streams[i]);
        cudaMallocAsync(&d_option_values_next, (run.n + 1) * sizeof(double), streams[i]);
        cudaMallocAsync(&st_buffer, (2 * run.n + 2) * sizeof(double), streams[i]);

        int fill_num_blocks = std::ceil((2 * run.n + 1) * 1.0 / num_threads);
        FUNC_NAME(fill_st_buffer_kernel)<<<fill_num_blocks, num_threads, 0, streams[i]>>>(st_buffer, run.S, run.K, u,
                                                                                           option_type_sign(run.type), run.n);

        int num_blocks = std::ceil((run.n + 1) * 1.0 / num_threads);
        FUNC_NAME(compute_first_layer_kernel)<<<num_blocks, num_threads, 0, streams[i]>>>(d_option_values, st_buffer, run.n);

        for (int level = run.n - 1; level >= 0; level--) {
            num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
            FUNC_NAME(compute_next_layer)<<<num_blocks, num_threads, 0, streams[i]>>>(
                d_option_values, d_option_values_next, st_buffer, up, down, level, run.n);
            std::swap(d_option_values, d_option_values_next);
        }

        d_option_values_ds.push_back(d_option_values);
        d_option_values_next_ds.push_back(d_option_values_next);
        st_buffer_ds.push_back(st_buffer);
    }

    std::vector<double> results(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        cudaMemcpyAsync(&results[i], d_option_values_ds[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
        cudaFreeAsync(d_option_values_ds[i], streams[i]);
        cudaFreeAsync(d_option_values_next_ds[i], streams[i]);
        cudaFreeAsync(st_buffer_ds[i], streams[i]);
    }

    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());

    for (size_t i = 0; i < streams.size(); ++i) cudaStreamDestroy(streams[i]);

    return results;
}
