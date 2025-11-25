#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"
#include "ds_float.cuh"

#define IMPL_NAME bkdstprcmp_xdovlpunroll_shuffle_trimotm_single
#define WARP_SIZE 32

__global__ void FUNC_NAME(fill_st_buffers_kernel)(double* __restrict__ st_buffer_bank0,
                                                  double* __restrict__ st_buffer_bank1,
                                                  const double S, const double K, const double u,
                                                  const int sign, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double u_pow_2_threadId = pow(u, (double)2 * threadId);
    double u_pow_minus_n = pow(u, (double)-n);

    // entry i stores value corresponding to exponent 2*i - n
    st_buffer_bank0[threadId] = fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n, -K), 0.0);

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[threadId] = fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n * u, -K), 0.0);
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel)(
    const double* __restrict__ layer_values_read, double* __restrict__ layer_values_write,
    const double* __restrict__ st_buffer_bank0, const double* __restrict__ st_buffer_bank1,
    const double up, const double down, const int level, const int n, const int upper_bound) {
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Use DS float for shared memory
    __shared__ ds_float warp_edges_layer_values_tile[NUM_WARPS + 1];

    int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    int tile_base = tile_stride * blockIdx.x;
    int node_id = tile_base + threadIdx.x;

    // Convert parameters to DS at kernel entry
    ds_float up_ds = double_to_ds(up);
    ds_float down_ds = double_to_ds(down);

    // Load and convert to DS
    ds_float val = double_to_ds(layer_values_read[node_id]);
    __syncwarp();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const double* st_buffer_bank = (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // Warp shuffle for DS
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];

        // DS arithmetic: hold = up * up_val + down * val
        ds_float down_mul_val = ds_mul(down_ds, val);
        ds_float hold = ds_fma(up_ds, up_val, down_mul_val);

        // Load exercise value and convert to DS
        int st_index = node_id + (n - current_level) / 2;
        ds_float exercise = double_to_ds(st_buffer_bank[st_index]);

        // DS max
        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        // Convert back to double for output
        layer_values_write[node_id] = ds_to_double(val);
    }
}

/*
    At each layer l exercise value of node i (from the bottom) is calculated with the following
    exponent: 2*i - l = 2 * (i + (n - l) / 2) - n           if (n - l) evenOption 3: Double-Single (DS) Arithmetic
Emulate FP64 using pairs of FP32 numbers. Each value stored as (high, low) where value ≈ high + low. Pros: Better precision than FP32, faster than FP64 on your GPU (2-4x overhead vs 64x)
Cons: Requires rewriting all arithmetic operations, more complex code
    the correspoding value is stored at st_buffer_bank0[i + (n - l) / 2]

    2*i - l = 2 * (i + (n - l - 1) / 2) - n + 1   if (n - l) odd
    the correspoding value is stored at st_buffer_bank1[i + (n - l) / 2]
*/
__global__ void FUNC_NAME(compute_next_layer_kernel)(const double* __restrict__ layer_values_read,
                                                     double* __restrict__ layer_values_write,
                                                     const double* __restrict__ st_buffer_bank0,
                                                     const double* __restrict__ st_buffer_bank1,
                                                     const double up, const double down,
                                                     const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double hold = up * layer_values_read[threadId + 1] + down * layer_values_read[threadId];
    const double* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    double exercise = st_buffer_bank[threadId + (n - level) / 2];
    layer_values_write[threadId] = fmax(hold, exercise);
}

int FUNC_NAME(search_bound)(const int n, const double S, const double K, const double u,
                            const int sign) {
    if (sign == 1) return n;

    int lower = 0;
    int upper = n;
    while (lower < upper) {
        int mid = lower + (upper - lower + 1) / 2;
        double S_mid_n = sign * (S * std::pow(u, mid * 2 - n) - K);
        if (S_mid_n < 0.)
            upper = mid - 1;
        else
            lower = mid;
    }
    return lower;
}

template <const Hyperparams& h>
double FUNC_NAME(vanilla_american_binomial_cuda)(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {
    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    const double delta_t = T / n;
    const double u = std::exp(sigma * std::sqrt(delta_t));
    const double d = 1.0 / u;
    const double p = (exp((r - q) * delta_t) - d) / (u - d);
    const double discount = std::exp(-r * delta_t);
    const double up = p * discount;
    const double down = (1.0 - p) * discount;
    const int sign = option_type_sign(type);

    double *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, (n + THREADS_PER_BLOCK) * sizeof(double));
    cudaMalloc(&layer_values_write_d, (n + THREADS_PER_BLOCK) * sizeof(double));

    double *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(double));
    cudaMalloc(&st_buffer_bank1_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(double));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(fill_st_buffers_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
        st_buffer_bank0_d, st_buffer_bank1_d, S, K, u, sign, n);

    // Layer n is the first n + 1 entries of st_buffer_bank0_d
    cudaMemcpy(layer_values_read_d, st_buffer_bank0_d, (n + 1) * sizeof(double),
               cudaMemcpyDeviceToDevice);

    int bound = FUNC_NAME(search_bound)(n, S, K, u, sign);
    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        int num_nodes = std::min(level, bound);
        num_blocks =
            std::ceil((num_nodes + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        FUNC_NAME(compute_next_layers_kernel)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                st_buffer_bank0_d, st_buffer_bank1_d, up, down,
                                                level, n, num_nodes);
        std::swap(layer_values_read_d, layer_values_write_d);
    }
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up,
            down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    cudaDeviceSynchronize();

    double value_h;
    cudaMemcpy(&value_h, layer_values_read_d, (1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);

    return value_h;
}

template double FUNC_NAME(
    vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

#ifdef DO_CARTESIAN_PRODUCT
#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM(  \
    ID, A, B, C, D, E, Y)                                                                    \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>( \
        const double S, const double K, const double T, const double r, const double sigma,  \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM,
               HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif
