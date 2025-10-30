#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define THREADS_PER_BLOCK 1024
#define NUM_OUTS_PER_THREAD 2

__global__ void fill_pricing_tile(double* __restrict__ buffer, const double S, const double K,
                                  const double u, const int sign, const int n) {
  int threadId = (blockIdx.x * blockDim.x + threadIdx.x);  // offset of result idx
  if (threadId > 2 * n) return;
  buffer[threadId] = max(sign * (S * pow(u, threadId - n) - K), 0.0);
}

__global__ void first_layer_kernel_tile(double* __restrict__ d_option_values,
                                        double* __restrict__ st_buffer, const int n) {
  int threadId = NUM_OUTS_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
  if (threadId > n) return;
#pragma unroll
  for (int i = 0; i < NUM_OUTS_PER_THREAD; i++) {
    d_option_values[threadId + i] = st_buffer[2 * (threadId + i) + 1];
  }
}

__global__ void vanilla_american_binomial_cuda_kernel_tile(
    double* __restrict__ d_option_values, double* __restrict__ d_option_values_next,
    double* __restrict__ st_buffer, const double prob_up, const double prob_down, const int level,
    const int n) {
  __shared__ double tile[THREADS_PER_BLOCK * NUM_OUTS_PER_THREAD + 1];
  int threadGlobalOffset = NUM_OUTS_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);

  // 0   1
  // 0 1 2 3
  for (int i = NUM_OUTS_PER_THREAD * threadIdx.x; i < THREADS_PER_BLOCK * NUM_OUTS_PER_THREAD + 1;
       i += THREADS_PER_BLOCK * NUM_OUTS_PER_THREAD) {
    int global_offset = NUM_OUTS_PER_THREAD * (blockIdx.x * blockDim.x);
    for (int j = 0; j < NUM_OUTS_PER_THREAD; j++)
      tile[i + j] = d_option_values[global_offset + i + j];
  }
  // wait for threads to finish setting up tile
  __syncthreads();

  if (threadGlobalOffset > level) return;
// threadId = (threadId % (level+1));
#pragma unroll
  for (int i = 0; i < NUM_OUTS_PER_THREAD && threadGlobalOffset + i <= level; i++) {
    double hold = prob_up * tile[NUM_OUTS_PER_THREAD * threadIdx.x + i + 1] +
                  prob_down * tile[NUM_OUTS_PER_THREAD * threadIdx.x + i];
    double exercise = st_buffer[2 * (threadGlobalOffset + i) - level + n];
    d_option_values_next[threadGlobalOffset + i] = max(hold, exercise);
  }
}

double vanilla_american_binomial_cuda_tile(const double S, const double K, const double T,
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

  const int thread_per_block = THREADS_PER_BLOCK;
  int num_blocks = std::ceil((n + 1) * 1.0 / (NUM_OUTS_PER_THREAD * thread_per_block));

  double *d_option_values, *d_option_values_next;
  cudaMalloc(&d_option_values, (n + 1) * sizeof(double));
  cudaMalloc(&d_option_values_next, (n + 1) * sizeof(double));
  double* st_buffer;
  cudaMalloc(&st_buffer, (2 * n + 1) * sizeof(double));

  int fill_num_blocks = std::ceil((2 * n + 1) * 1.0 / thread_per_block);
  fill_pricing_tile<<<fill_num_blocks, thread_per_block>>>(st_buffer, S, K, u, sign, n);

  first_layer_kernel_tile<<<num_blocks, thread_per_block>>>(d_option_values, st_buffer, n);
  for (int level = n - 1; level >= 0; level--) {
    num_blocks = std::ceil((level + 1) * 1.0 / (thread_per_block * NUM_OUTS_PER_THREAD));
    vanilla_american_binomial_cuda_kernel_tile<<<num_blocks, thread_per_block>>>(
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
