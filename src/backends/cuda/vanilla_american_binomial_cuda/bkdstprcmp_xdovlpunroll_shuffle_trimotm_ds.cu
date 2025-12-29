#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>
#include <map>

#include "backends/cuda/ds_float.cuh"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

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
    st_buffer_bank0[threadId] =
        double_to_ds(fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n, -K), 0.0));

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[threadId] =
        double_to_ds(fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n * u, -K), 0.0));
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float up_ds, const ds_float down_ds, const int level, const int n,
    const int upper_bound) {
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Use DS float for shared memory4110.889
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
        const ds_float* st_buffer_bank =
            (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // Warp shuffle for DS
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS arithmetic: hold = up * up_val + down * val
        ds_float hold = ds_add_opt(ds_mul_opt(up_ds, up_val), ds_mul_opt(down_ds, val));

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
    ds_float hold = ds_add_opt(ds_mul_opt(up_ds, layer_values_read[threadId + 1]),
                               ds_mul_opt(down_ds, layer_values_read[threadId]));

    const ds_float* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    ds_float exercise = st_buffer_bank[threadId + (n - level) / 2];
    layer_values_write[threadId] = ds_max(hold, exercise);
}

int FUNC_NAME(search_bound)(const int n, const double S, const double K, const double u,
                            const int sign) {
    if (sign == 1) return n;

    // Scheduler: pick a compile-time UNROLL_FACTOR based on the current `level`
    // and run the corresponding templated kernel in a tight loop. This mirrors
    // the logic in `get_unroll_factor_for_n` but avoids attempting to assign
    // to a constexpr template parameter at runtime.
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
    cudaMallocAsync(&st_buffer_bank0_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float),
                    0);
    cudaMallocAsync(&st_buffer_bank1_d, (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float),
                    0);

    // Initialize st_buffer_banks to zero
    cudaMemsetAsync(st_buffer_bank0_d, 0,
                    (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float));
    cudaMemsetAsync(st_buffer_bank1_d, 0,
                    (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * sizeof(ds_float));

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
                                                st_buffer_bank0_d, st_buffer_bank1_d, up_ds,
                                                down_ds, level, n, num_nodes);
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

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS( \
    ID, A, B, C, D, E, Y)                                                                      \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>(   \
        const double S, const double K, const double T, const double r, const double sigma,    \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(
    PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS,
    HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(fill_st_buffers_kernel_batch)(
    ds_float* __restrict__ st_buffer_bank0, ds_float* __restrict__ st_buffer_bank1,
    const double* __restrict__ S, const double* __restrict__ K, const double* __restrict__ u,
    const int* __restrict__ sign, const int n, ds_float* __restrict__ layer_values) {
    const int option_idx = blockIdx.y;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculation logic uses double for the power steps
    const double u_pow_2_threadId = pow(u[option_idx], (double)2 * threadId);
    const double u_pow_minus_n = pow(u[option_idx], (double)-n);

    const int st_index = option_idx * (n + THREADS_PER_BLOCK + UNROLL_FACTOR) + threadId;
    const int layer_index = option_idx * (n + THREADS_PER_BLOCK) + threadId;

    // entry i stores value corresponding to exponent 2*i - n
    // Compute in double -> Convert to DS -> Store
    ds_float val0 = double_to_ds(fmax(
        sign[option_idx] * fma(S[option_idx], u_pow_2_threadId * u_pow_minus_n, -K[option_idx]),
        0.0));
    st_buffer_bank0[st_index] = val0;
    layer_values[layer_index] = val0;

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[st_index] = double_to_ds(fmax(
        sign[option_idx] *
            fma(S[option_idx], u_pow_2_threadId * u_pow_minus_n * u[option_idx], -K[option_idx]),
        0.0));
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel_batch_schedule)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float* __restrict__ up, const ds_float* __restrict__ down, const int level,
    const int n, const int* __restrict__ upper_bound, const int MAX_UNROLL_FACTOR) {
    const int option_idx = blockIdx.y;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Offsets
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + MAX_UNROLL_FACTOR) * option_idx;

    // Shared memory needs to be ds_float
    __shared__ ds_float warp_edges_layer_values_tile[NUM_WARPS + 1];

    const int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    const int tile_base = tile_stride * blockIdx.x;
    const int node_id = tile_base + threadIdx.x;

    if (node_id > upper_bound[option_idx]) return;

    // Load DS value
    ds_float val = layer_values_read[base_layer_values + node_id];
    __syncwarp();

    // Cache up/down for this option index in registers
    const ds_float my_up = up[option_idx];
    const ds_float my_down = down[option_idx];

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const ds_float* st_buffer_bank =
            (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // DS Shuffle
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS Math: hold = up * up_val + down * val
        ds_float hold = ds_add_two_mults_streamlined(my_up, up_val, my_down, val);

        int st_index = node_id + (n - current_level) / 2;
        ds_float exercise = st_buffer_bank[base_st_buffer + st_index];
        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_layer_values + node_id] = val;
    }
}
template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(test_compute_next_layers_kernel_batch)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float* __restrict__ up, const ds_float* __restrict__ down, const int level,
    const int n, const int* __restrict__ upper_bound) {
    const int option_idx = blockIdx.y;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Offsets
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * option_idx;

    // Shared memory needs to be ds_float
    __shared__ ds_float warp_edges_layer_values_tile[NUM_WARPS + 1];

    const int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    const int tile_base = tile_stride * blockIdx.x;
    const int node_id = tile_base + threadIdx.x;

    if (node_id > upper_bound[option_idx]) return;

    // Load DS value
    ds_float val = layer_values_read[base_layer_values + node_id];
    __syncwarp();

    // Cache up/down for this option index in registers
    const ds_float my_up = up[option_idx];
    const ds_float my_down = down[option_idx];

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const ds_float* st_buffer_bank =
            (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // DS Shuffle
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS Math: hold = up * up_val + down * val
        // ds_float hold = ds_add_opt(ds_mul_opt_fused(my_up, up_val), ds_mul_opt_fused(my_down,
        // val));
        ds_float hold = ds_add_two_mults_streamlined(my_up, up_val, my_down, val);

        int st_index = node_id + (n - current_level) / 2;
        ds_float exercise = st_buffer_bank[base_st_buffer + st_index];
        //
        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_layer_values + node_id] = val;
    }
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel_batch)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float* __restrict__ up, const ds_float* __restrict__ down, const int level,
    const int n, const int* __restrict__ upper_bound) {
    const int option_idx = blockIdx.y;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Offsets
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * option_idx;

    // Shared memory needs to be ds_float
    __shared__ ds_float warp_edges_layer_values_tile[NUM_WARPS + 1];

    const int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    const int tile_base = tile_stride * blockIdx.x;
    const int node_id = tile_base + threadIdx.x;

    if (node_id > upper_bound[option_idx]) return;

    // Load DS value
    ds_float val = layer_values_read[base_layer_values + node_id];
    __syncwarp();

    // Cache up/down for this option index in registers
    const ds_float my_up = up[option_idx];
    const ds_float my_down = down[option_idx];

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const ds_float* st_buffer_bank =
            (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        // DS Shuffle
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS Math: hold = up * up_val + down * val
        ds_float hold = ds_add_opt(ds_mul_opt(my_up, up_val), ds_mul_opt(my_down, val));
        // ds_add_two_mults_opt(ds_float my_up, ds_float up_val, ds_float my_down, ds_float val)

        int st_index = node_id + (n - current_level) / 2;
        ds_float exercise = st_buffer_bank[base_st_buffer + st_index];
        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_layer_values + node_id] = val;
    }
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layer_kernel_batch)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ st_buffer_bank0, const ds_float* __restrict__ st_buffer_bank1,
    const ds_float* __restrict__ up, const ds_float* __restrict__ down, const int level,
    const int n) {
    const int option_idx = blockIdx.y;
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * option_idx;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // DS Math
    ds_float hold =
        ds_add_opt(ds_mul_opt(up[option_idx], layer_values_read[base_layer_values + threadId + 1]),
                   ds_mul_opt(down[option_idx], layer_values_read[base_layer_values + threadId]));

    const ds_float* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

    ds_float exercise = st_buffer_bank[base_st_buffer + threadId + (n - level) / 2];

    layer_values_write[base_layer_values + threadId] = ds_max(hold, exercise);
}

template <const int THREADS_PER_BLOCK>
__global__ void FUNC_NAME(copy_final_value)(const ds_float* __restrict__ layer_values_read,
                                            double* __restrict__ out, const int n,
                                            const size_t num_runs) {
    const int option_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (option_idx >= num_runs) return;

    // Read DS, convert to double, write to output
    ds_float res_ds = layer_values_read[option_idx * (n + THREADS_PER_BLOCK)];
    out[option_idx] = ds_to_double(res_ds);
}

void FUNC_NAME(vanilla_american_binomial_cuda_batch_scheduler)(std::vector<PricingInput>& runs,
                                                               std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int MAX_UNROLL_FACTOR = 128;

    // Host buffers
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    // Note: up/down are DS on host
    std::vector<ds_float> h_up_ds(num_runs), h_down_ds(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);
    std::vector<int> h_bound(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;

        int bound = FUNC_NAME(search_bound)(run.n, run.S, run.K, u, option_type_sign(run.type));
        h_bound[i] = bound;
        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);

        // Compute double values, convert to DS on Host
        h_up_ds[i] = double_to_ds(p * risk_free_rate);
        h_down_ds[i] = double_to_ds(one_minus_p * risk_free_rate);
    }

    // Device allocation
    double *d_S, *d_K, *d_u;
    int *d_bound, *d_n_arr, *d_sign;
    ds_float *d_up, *d_down;  // Pointers to DS arrays
    double* d_out;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(ds_float));
    cudaMalloc(&d_down, num_runs * sizeof(ds_float));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_bound, num_runs * sizeof(int));
    cudaMalloc(&d_out, num_runs * sizeof(double));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);

    // Copy DS arrays
    cudaMemcpy(d_up, h_up_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bound, h_bound.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    int n = runs[0].n;

    const int layer_size = num_runs * (n + THREADS_PER_BLOCK);
    ds_float *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, layer_size * sizeof(ds_float));
    cudaMalloc(&layer_values_write_d, layer_size * sizeof(ds_float));
    // Initialize to zero to be safe
    cudaMemsetAsync(layer_values_read_d, 0, layer_size * sizeof(ds_float));

    const int buffer_size = num_runs * (n + THREADS_PER_BLOCK + MAX_UNROLL_FACTOR);
    ds_float *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, buffer_size * sizeof(ds_float));
    cudaMalloc(&st_buffer_bank1_d, buffer_size * sizeof(ds_float));

    // 1. Fill buffers
    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);

    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, MAX_UNROLL_FACTOR>
        <<<num_blocks_2d, THREADS_PER_BLOCK>>>(st_buffer_bank0_d, st_buffer_bank1_d, d_S, d_K, d_u,
                                               d_sign, n, layer_values_read_d);
    std::string kernel_label = "Kernel: compute_next_layers_kernel_batch_schedule";
    int level = n,lastU = -1;

    
    auto get_current_unrool_factor = [](int current_level) -> int {
        if (current_level >= (1 << 20)) {
            return 16;
        } else if (current_level >= (1 << 14)) {
            return 24;
        } else if (current_level >= (1 << 12)) {
            return 32;
        } else if (current_level >= 64) {
            return 64;
        } else if (current_level >= (32)) {
            return 32;
        } else {
            return 1;
        }
    };

    // Launch the correct templated kernel based on the chosen unroll factor.
    for (; level > 0; ) {
        int U = get_current_unrool_factor(level);

        if (lastU!= U) {
            nvtxRangePushA(kernel_label.c_str());
            cudaProfilerStart();
        }
        num_blocks = std::ceil((level) * 1.0 / (THREADS_PER_BLOCK - U));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        

        switch (U) {
            case 1:
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 1>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR);
                break;
            case 16:
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 16>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR);
                break;
            case 24:
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 24>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR);
                break;
            case 32:
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 32>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR);
                break;
            case 64:
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 64>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR);
                break;
            default:
                // Fallback to U=1 if an unsupported unroll factor is requested.
                FUNC_NAME(compute_next_layers_kernel_batch_schedule)<THREADS_PER_BLOCK, 1>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d,
                        d_up, d_down, level - 1, n, d_bound, MAX_UNROLL_FACTOR);
                U = 1;
                break;
        }

        std::swap(layer_values_read_d, layer_values_write_d);
        level -= U;
        if (lastU!= U) {
            cudaDeviceSynchronize();
            cudaProfilerStop();
            nvtxRangePop();
            lastU = U;
        }
    }

    num_blocks = std::ceil(num_runs * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(copy_final_value)<THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, d_out, n, num_runs);

    cudaMemcpy(out.data(), d_out, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_bound);
    cudaFree(d_out);
    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);
    checkCuda(cudaGetLastError());
}

template <const Hyperparams& h>
void FUNC_NAME(test_vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                          std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    // Host buffers
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    // Note: up/down are DS on host
    std::vector<ds_float> h_up_ds(num_runs), h_down_ds(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);
    std::vector<int> h_bound(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;

        int bound = FUNC_NAME(search_bound)(run.n, run.S, run.K, u, option_type_sign(run.type));
        h_bound[i] = bound;
        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);

        // Compute double values, convert to DS on Host
        h_up_ds[i] = double_to_ds(p * risk_free_rate);
        h_down_ds[i] = double_to_ds(one_minus_p * risk_free_rate);
    }

    // Device allocation
    double *d_S, *d_K, *d_u;
    int *d_bound, *d_n_arr, *d_sign;
    ds_float *d_up, *d_down;  // Pointers to DS arrays
    double* d_out;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(ds_float));
    cudaMalloc(&d_down, num_runs * sizeof(ds_float));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_bound, num_runs * sizeof(int));
    cudaMalloc(&d_out, num_runs * sizeof(double));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_up, h_up_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bound, h_bound.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    int n = runs[0].n;

    const int layer_size = num_runs * (n + THREADS_PER_BLOCK);
    ds_float *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, layer_size * sizeof(ds_float));
    cudaMalloc(&layer_values_write_d, layer_size * sizeof(ds_float));
    cudaMemsetAsync(layer_values_read_d, 0, layer_size * sizeof(ds_float));

    const int buffer_size = num_runs * (n + THREADS_PER_BLOCK + UNROLL_FACTOR);
    ds_float *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, buffer_size * sizeof(ds_float));
    cudaMalloc(&st_buffer_bank1_d, buffer_size * sizeof(ds_float));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);

    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
        <<<num_blocks_2d, THREADS_PER_BLOCK>>>(st_buffer_bank0_d, st_buffer_bank1_d, d_S, d_K, d_u,
                                               d_sign, n, layer_values_read_d);

    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(test_compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n, d_bound);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(compute_next_layer_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    num_blocks = std::ceil(num_runs * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(copy_final_value)<THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, d_out, n, num_runs);

    cudaMemcpy(out.data(), d_out, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_bound);
    cudaFree(d_out);
    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);
    checkCuda(cudaGetLastError());
}

template <const Hyperparams& h>
void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    // Host buffers
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    // Note: up/down are DS on host
    std::vector<ds_float> h_up_ds(num_runs), h_down_ds(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);
    std::vector<int> h_bound(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;

        int bound = FUNC_NAME(search_bound)(run.n, run.S, run.K, u, option_type_sign(run.type));
        h_bound[i] = bound;
        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);

        // Compute double values, convert to DS on Host
        h_up_ds[i] = double_to_ds(p * risk_free_rate);
        h_down_ds[i] = double_to_ds(one_minus_p * risk_free_rate);
    }

    // Device allocation
    double *d_S, *d_K, *d_u;
    int *d_bound, *d_n_arr, *d_sign;
    ds_float *d_up, *d_down;  // Pointers to DS arrays
    double* d_out;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(ds_float));
    cudaMalloc(&d_down, num_runs * sizeof(ds_float));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_bound, num_runs * sizeof(int));
    cudaMalloc(&d_out, num_runs * sizeof(double));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_up, h_up_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bound, h_bound.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    int n = runs[0].n;

    const int layer_size = num_runs * (n + THREADS_PER_BLOCK);
    ds_float *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, layer_size * sizeof(ds_float));
    cudaMalloc(&layer_values_write_d, layer_size * sizeof(ds_float));
    cudaMemsetAsync(layer_values_read_d, 0, layer_size * sizeof(ds_float));

    const int buffer_size = num_runs * (n + THREADS_PER_BLOCK + UNROLL_FACTOR);
    ds_float *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, buffer_size * sizeof(ds_float));
    cudaMalloc(&st_buffer_bank1_d, buffer_size * sizeof(ds_float));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);

    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
        <<<num_blocks_2d, THREADS_PER_BLOCK>>>(st_buffer_bank0_d, st_buffer_bank1_d, d_S, d_K, d_u,
                                               d_sign, n, layer_values_read_d);

    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n, d_bound);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(compute_next_layer_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    num_blocks = std::ceil(num_runs * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(copy_final_value)<THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, d_out, n, num_runs);

    cudaMemcpy(out.data(), d_out, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_bound);
    cudaFree(d_out);
    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);
    checkCuda(cudaGetLastError());
}

template <const Hyperparams& h>
void FUNC_NAME(vanilla_american_binomial_cuda_batch_search)(std::vector<PricingInput>& runs,
                                                            std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;
    constexpr int FROM_N = h.OUTPUTS_PER_THREAD;
    int TILL_N = INT_MIN;
    if (TILL_N > 2) {
        for (auto& run : runs) {
            run.n = pow(2, FROM_N);
        }
        TILL_N = pow(2, FROM_N - 1);
    }

    // Host buffers
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    // Note: up/down are DS on host
    std::vector<ds_float> h_up_ds(num_runs), h_down_ds(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);
    std::vector<int> h_bound(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;

        int bound = FUNC_NAME(search_bound)(run.n, run.S, run.K, u, option_type_sign(run.type));
        h_bound[i] = bound;
        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);

        // Compute double values, convert to DS on Host
        h_up_ds[i] = double_to_ds(p * risk_free_rate);
        h_down_ds[i] = double_to_ds(one_minus_p * risk_free_rate);
    }

    // Device allocation
    double *d_S, *d_K, *d_u;
    int *d_bound, *d_n_arr, *d_sign;
    ds_float *d_up, *d_down;  // Pointers to DS arrays
    double* d_out;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(ds_float));
    cudaMalloc(&d_down, num_runs * sizeof(ds_float));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_bound, num_runs * sizeof(int));
    cudaMalloc(&d_out, num_runs * sizeof(double));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);

    // Copy DS arrays
    cudaMemcpy(d_up, h_up_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down_ds.data(), num_runs * sizeof(ds_float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bound, h_bound.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    int n = runs[0].n;

    // Allocate buffers using ds_float size
    const int layer_size = num_runs * (n + THREADS_PER_BLOCK);
    ds_float *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, layer_size * sizeof(ds_float));
    cudaMalloc(&layer_values_write_d, layer_size * sizeof(ds_float));
    // Initialize to zero to be safe
    cudaMemsetAsync(layer_values_read_d, 0, layer_size * sizeof(ds_float));

    const int buffer_size = num_runs * (n + THREADS_PER_BLOCK + UNROLL_FACTOR);
    ds_float *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, buffer_size * sizeof(ds_float));
    cudaMalloc(&st_buffer_bank1_d, buffer_size * sizeof(ds_float));

    // 1. Fill buffers
    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);

    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
        <<<num_blocks_2d, THREADS_PER_BLOCK>>>(st_buffer_bank0_d, st_buffer_bank1_d, d_S, d_K, d_u,
                                               d_sign, n, layer_values_read_d);

    // 2. Unrolled compute loop
    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        if (level < TILL_N) break;
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n, d_bound);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    // 3. Remainder compute loop
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        if (level < TILL_N) break;
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

        FUNC_NAME(compute_next_layer_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                        st_buffer_bank0_d, st_buffer_bank1_d, d_up,
                                                        d_down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    // 4. Copy final values
    num_blocks = std::ceil(num_runs * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(copy_final_value)<THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, d_out, n, num_runs);

    cudaMemcpy(out.data(), d_out, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_bound);
    cudaFree(d_out);
    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);
    checkCuda(cudaGetLastError());
}
double FUNC_NAME(vanilla_american_binomial_cuda_scheduler)(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type){
    std::vector<PricingInput> runs =  { PricingInput(S, K, T, r, sigma, q, n, type) };
    std::vector<double> out{0};

    vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds(runs, out);
    return out[0];
}


template void FUNC_NAME(test_vanilla_american_binomial_cuda_batch)<
    DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(std::vector<PricingInput>& runs,
                                                             std::vector<double>& out);

template void FUNC_NAME(
    vanilla_american_binomial_cuda_batch)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(
    std::vector<PricingInput>& runs, std::vector<double>& out);


template void FUNC_NAME(vanilla_american_binomial_cuda_batch_search)<
    DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(std::vector<PricingInput>& runs,
                                                             std::vector<double>& out);

#ifdef DO_CARTESIAN_PRODUCT

#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BATCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS
#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BATCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS( \
    ID, A, B, C, D, E, Y)                                                                            \
    template void FUNC_NAME(vanilla_american_binomial_cuda_batch)<GRID_SEARCH_HYPERPARAMS_##ID>(     \
        std::vector<PricingInput> & runs, std::vector<double> & out);
APPLY_FUNCTION(
    PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BATCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS,
    HYPERPARAMS_CART_PRODUCT, NULL)

#endif

#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BATCH_SEARCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS
#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BATCH_SEARCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS( \
    ID, A, B, C, D, E, Y)                                                                                   \
    template void FUNC_NAME(                                                                                \
        vanilla_american_binomial_cuda_batch_search)<GRID_SEARCH_HYPERPARAMS_##ID>(                         \
        std::vector<PricingInput> & runs, std::vector<double> & out);
APPLY_FUNCTION(
    PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BATCH_SEARCH_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS,
    HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif

