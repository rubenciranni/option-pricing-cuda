#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define IMPL_NAME precomputed_payoffs

__global__ void FUNC_NAME(compute_next_layer_kernel)(double* d_option_values,
                                                     double* d_option_values_next, const double up,
                                                     const double down,
                                                     const double* d_payoff_values, const int level,
                                                     const int n) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id > level) return;
    int exponent = 2 * thread_id - level;
    double hold = up * d_option_values[thread_id + 1] + down * d_option_values[thread_id];
    double exercise = max(d_payoff_values[exponent + n], 0.0);
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

    // Precompute all possible payoff values: sign * (ST - K)
    // We need values for exponents from -n to +n (total 2n+1 values)
    const int num_values = 2 * n + 1;
    double* h_payoff_values = new double[num_values];

    // Compute center value (u^0 = 1.0)
    h_payoff_values[n] = sign * (S - K);

    // Compute positive powers and payoffs simultaneously: S*u^1, S*u^2, ...,
    // S*u^n
    double S_up = S;
    for (int i = 1; i <= n; i++) {
        S_up *= u;
        h_payoff_values[n + i] = sign * (S_up - K);
    }

    // Compute negative powers and payoffs simultaneously: S*u^(-1), S*u^(-2),
    // ..., S*u^(-n)
    double S_down = S;
    for (int i = 1; i <= n; i++) {
        S_down *= d;
        h_payoff_values[n - i] = sign * (S_down - K);
    }

    // Compute first layer on CPU (no need for GPU kernel)
    double* h_option_values = new double[n + 1];
    for (int i = 0; i <= n; i++) {
        int exponent = 2 * i - n + 1;
        double payoff = h_payoff_values[exponent + n];
        h_option_values[i] = payoff > 0.0 ? payoff : 0.0;
    }

    // Copy precomputed payoffs to device
    double* d_payoff_values;
    cudaMalloc(&d_payoff_values, num_values * sizeof(double));
    cudaMemcpy(d_payoff_values, h_payoff_values, num_values * sizeof(double),
               cudaMemcpyHostToDevice);
    delete[] h_payoff_values;

    const int num_threads = 1024;

    double *d_option_values, *d_option_values_next;
    cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
    cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));

    // Copy first layer from CPU to GPU
    cudaMemcpy(d_option_values, h_option_values, (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_option_values;

    // Backward induction on GPU
    for (int level = n - 1; level >= 0; level--) {
        int num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, num_threads>>>(
            d_option_values, d_option_values_next, up, down, d_payoff_values, level, n);
        std::swap(d_option_values, d_option_values_next);
    }

    cudaDeviceSynchronize();
    double h_s_store;
    cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_option_values);
    cudaFree(d_option_values_next);
    cudaFree(d_payoff_values);

    return h_s_store;
}
