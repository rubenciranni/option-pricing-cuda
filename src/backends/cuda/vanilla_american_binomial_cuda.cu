#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

// Optimized kernels with precomputed stock prices
__global__ void first_layer_kernel_precomputed(double* d_option_values, int level,
                                               const double* d_stock_prices, double K, const int sign,
                                               const int n) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id > level) return;
  int exponent = 2 * thread_id - level + 1;
  double ST = d_stock_prices[exponent + n];  // Lookup instead of pow()
  d_option_values[thread_id] = max(0.0, sign * (ST - K));
}

__global__ void vanilla_american_binomial_cuda_kernel_precomputed(
    double* d_option_values, double* d_option_values_next, const double K, const double up,
    const double down, const double* d_stock_prices, const int level, const int sign, const int n) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id > level) return;
  int exponent = 2 * thread_id - level;
  double ST = d_stock_prices[exponent + n];  // Lookup instead of pow()
  double hold = up * d_option_values[thread_id + 1] + down * d_option_values[thread_id];
  double exercise = max(sign * (ST - K), 0.0);
  d_option_values_next[thread_id] = max(hold, exercise);
}

// Original kernels (kept for benchmarking)
__global__ void first_layer_kernel(double* d_option_values, int level, double S, double u, double K,
                                   const int sign) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id > level) return;
  double ST = S * pow(u, 2 * thread_id - level + 1);
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

double vanilla_american_binomial_cuda_precomputed(const double S, const double K, const double T,
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

  // Precompute all possible stock prices on host
  // We need prices for exponents from -n to +n (total 2n+1 values)
  const int num_prices = 2 * n + 1;
  double* h_stock_prices = new double[num_prices];

  // Compute S * u^i for i = -n to n
  for (int i = -n; i <= n; i++) {
    h_stock_prices[i + n] = S * std::pow(u, i);
  }

  // Copy precomputed prices to device
  double* d_stock_prices;
  cudaMalloc(&d_stock_prices, num_prices * sizeof(double));
  cudaMemcpy(d_stock_prices, h_stock_prices, num_prices * sizeof(double), cudaMemcpyHostToDevice);
  delete[] h_stock_prices;

  const int num_threads = 1024;
  int num_blocks = std::ceil((n + 1) * 1.0 / num_threads);

  double *d_option_values, *d_option_values_next;
  cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
  cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));

  // Use optimized kernels with precomputed prices
  first_layer_kernel_precomputed<<<num_blocks, num_threads>>>(d_option_values, n, d_stock_prices, K,
                                                                sign, n);
  for (int level = n - 1; level >= 0; level--) {
    num_blocks = std::ceil((level + 1) * 1.0 / num_threads);
    vanilla_american_binomial_cuda_kernel_precomputed<<<num_blocks, num_threads>>>(
        d_option_values, d_option_values_next, K, up, down, d_stock_prices, level, sign, n);
    std::swap(d_option_values, d_option_values_next);
  }

  cudaDeviceSynchronize();
  double h_s_store;
  cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_option_values);
  cudaFree(d_option_values_next);
  cudaFree(d_stock_prices);

  return h_s_store;
}
