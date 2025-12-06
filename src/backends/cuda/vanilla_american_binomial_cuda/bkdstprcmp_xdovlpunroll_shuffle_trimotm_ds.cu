#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"
#include "ds_float.cuh"

#define IMPL_NAME bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds
#define WARP_SIZE 32

__global__ void FUNC_NAME(fill_st_buffers_kernel)(ds_float* __restrict__ st_buffer_bank0,
                                                  ds_float* __restrict__ st_buffer_bank1,
                                                  const double S, const double K, const double u,
                                                  const int sign, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double u_pow_2_threadId = pow(u, (double)2 * threadId);
    double u_pow_minus_n = pow(u, (double)-n);

    // entry i stores value corresponding to exponent 2*i - n
    // Compute in double, convert once to DS
    st_buffer_bank0[threadId] = double_to_ds(
        fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n, -K), 0.0)
    );

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[threadId] = double_to_ds(
        fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n * u, -K), 0.0)
    );
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float up_ds, const ds_float down_ds, const int level, const int n, const int upper_bound) {
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

    // Load directly as DS (no conversion needed)
    ds_float val = layer_values_read[node_id];
    __syncwarp();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const ds_float* st_buffer_bank = (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // Warp shuffle for DS
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS arithmetic: hold = up * up_val + down * val
        ds_float hold = ds_add_opt(
            ds_mul_opt(up_ds, up_val),
            ds_mul_opt(down_ds, val)
        );

        // Load exercise value directly as DS (no conversion)
        int st_index = node_id + (n - current_level) / 2;
        ds_float exercise = st_buffer_bank[st_index];

        // DS max
        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        // Write DS directly (no conversion)
        layer_values_write[node_id] = val;
    }
}

__global__ void FUNC_NAME(compute_next_layer_kernel)(const ds_float* __restrict__ layer_values_read,
                                                     ds_float* __restrict__ layer_values_write,
                                                     const ds_float* __restrict__ st_buffer_bank0,
                                                     const ds_float* __restrict__ st_buffer_bank1,
                                                     const ds_float up_ds, const ds_float down_ds,
                                                     const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // DS arithmetic: ds_mul and ds_add already renormalize internally
    ds_float hold = ds_add_opt(
        ds_mul_opt(up_ds, layer_values_read[threadId + 1]),
        ds_mul_opt(down_ds, layer_values_read[threadId])
    );

    const ds_float* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    ds_float exercise = st_buffer_bank[threadId + (n - level) / 2];
    layer_values_write[threadId] = ds_max(hold, exercise);
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

    // Convert constants to DS once on host
    ds_float up_ds = double_to_ds(up);
    ds_float down_ds = double_to_ds(down);

    // Allocate DS float buffers
    ds_float *layer_values_read_d, *layer_values_write_d;
    cudaMallocAsync(&layer_values_read_d, (n + THREADS_PER_BLOCK) * sizeof(ds_float), 0);
    cudaMallocAsync(&layer_values_write_d, (n + THREADS_PER_BLOCK) * sizeof(ds_float), 0);

    // Initialize buffers to zero to prevent non-determinism from uninitialized memory
    cudaMemsetAsync(layer_values_read_d, 0, (n + THREADS_PER_BLOCK) * sizeof(ds_float));
    cudaMemsetAsync(layer_values_write_d, 0, (n + THREADS_PER_BLOCK) * sizeof(ds_float));

    ds_float *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMallocAsync(&st_buffer_bank0_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float), 0);
    cudaMallocAsync(&st_buffer_bank1_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float), 0);

    // Initialize st_buffer_banks to zero
    cudaMemsetAsync(st_buffer_bank0_d, 0, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float));
    cudaMemsetAsync(st_buffer_bank1_d, 0, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(fill_st_buffers_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
        st_buffer_bank0_d, st_buffer_bank1_d, S, K, u, sign, n);

    // Layer n is the first n + 1 entries of st_buffer_bank0_d
    cudaMemcpy(layer_values_read_d, st_buffer_bank0_d, (n + 1) * sizeof(ds_float),
               cudaMemcpyDeviceToDevice);

    int bound = FUNC_NAME(search_bound)(n, S, K, u, sign);
    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        int num_nodes = std::min(level, bound);
        num_blocks =
            std::ceil((num_nodes + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        FUNC_NAME(compute_next_layers_kernel)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                st_buffer_bank0_d, st_buffer_bank1_d, up_ds, down_ds,
                                                level, n, num_nodes);
        std::swap(layer_values_read_d, layer_values_write_d);
    }
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        FUNC_NAME(compute_next_layer_kernel)<<<num_blocks, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, up_ds,
            down_ds, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    cudaDeviceSynchronize();

    // Copy final DS result and convert once on host
    ds_float value_ds_h;
    cudaMemcpy(&value_ds_h, layer_values_read_d, sizeof(ds_float), cudaMemcpyDeviceToHost);
    double value_h = ds_to_double(value_ds_h);

    cudaFreeAsync(layer_values_read_d, 0);
    cudaFreeAsync(layer_values_write_d, 0);
    cudaFreeAsync(st_buffer_bank0_d, 0);
    cudaFreeAsync(st_buffer_bank1_d, 0);

    return value_h;
}

template double FUNC_NAME(
    vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

#ifdef DO_CARTESIAN_PRODUCT
#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS(  \
    ID, A, B, C, D, E, Y)                                                                           \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>(        \
        const double S, const double K, const double T, const double r, const double sigma,         \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS,
               HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif