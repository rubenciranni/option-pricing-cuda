#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

__global__ void fill_pricing(double* __restrict__ buffer, const double S, const double K,
                             const double u, const int sign, const int n) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  threadId = (threadId % (2 * n + 1));
  // st_buffer[0] = pow(u, -n)
  //  buffer[threadId] = max(sign*(S * pow(u, threadId-n) - K), 0.0);
  buffer[((threadId - n) & 1) * (n + 1) + (threadId) / 2] =
      max(sign * (S * pow(u, threadId - n) - K), 0.0);
}

__global__ void first_layer_kernel(double* d_option_values, double* st_buffer, const int n) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  // threadId = (threadId % (n+1));
  if (threadId > n) return;
  int exp = 2 * threadId + 1;
  // d_option_values[threadId] = st_buffer[2*threadId+1];
  d_option_values[threadId] = st_buffer[((exp - n) & 1) * (n + 1) + exp / 2];
}

__global__ void vanilla_american_binomial_cuda_kernel(double* d_option_values,
                                                      double* d_option_values_next,
                                                      double* st_buffer, const double prob_up,
                                                      const double prob_down, const int level,
                                                      const int n) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  // threadId = (threadId % (level+1));
  if (threadId > level) return;
  //   double ST = S * pow(u, 2.0 * thread_id - level);
  double hold = prob_up * d_option_values[threadId + 1] + prob_down * d_option_values[threadId];
  // double exercise = st_buffer[2*threadId - level+n];
  int exp = 2 * threadId - level;
  double exercise = st_buffer[(exp & 1) * (n + 1) + (exp + n) / 2];
  d_option_values_next[threadId] = max(hold, exercise);
}

double vanilla_american_binomial_cuda_fill(const double S, const double K, const double T,
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

  const int thread_per_block = 1024;
  int num_blocks = std::ceil((n + 1) * 1.0 / thread_per_block);

  double *d_option_values, *d_option_values_next;
  cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
  cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
  double* st_buffer;
  cudaMalloc(&st_buffer, (2 * n + 2) * sizeof(double));

  int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / thread_per_block);
  fill_pricing<<<fill_num_blocks, thread_per_block>>>(st_buffer, S, K, u, sign, n);

  first_layer_kernel<<<num_blocks, thread_per_block>>>(d_option_values, st_buffer, n);
  for (int level = n - 1; level >= 0; level--) {
    num_blocks = std::ceil((level + 1) * 1.0 / thread_per_block);
    vanilla_american_binomial_cuda_kernel<<<num_blocks, thread_per_block>>>(
        d_option_values, d_option_values_next, st_buffer, up, down, level, n);
    std::swap(d_option_values, d_option_values_next);
  }
  cudaDeviceSynchronize();
  double h_s_store;
  cudaMemcpy(&h_s_store, d_option_values, (1) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_option_values);
  cudaFree(d_option_values_next);
  cudaFree(st_buffer);
  return h_s_store;
}
