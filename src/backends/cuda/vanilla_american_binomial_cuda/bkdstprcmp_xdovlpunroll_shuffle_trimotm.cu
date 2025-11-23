#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

#define IMPL_NAME bkdstprcmp_xdovlpunroll_shuffle_trimotm
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
__global__ void FUNC_NAME(fill_st_buffers_kernel_batch)(
    double* __restrict__ st_buffer_bank0, double* __restrict__ st_buffer_bank1,
    const double* __restrict__ S, const double* __restrict__ K, const double* __restrict__ u,
    const int* __restrict__ sign, const int  n, double* __restrict__ layer_values) {
    const int option_idx = blockIdx.y;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    const double u_pow_2_threadId = pow(u[option_idx], (double)2 * threadId);
    const double u_pow_minus_n = pow(u[option_idx], (double)-n);

    const int st_index =
        option_idx * (n + THREADS_PER_BLOCK + UNROLL_FACTOR) + threadId;
    const int layer_index = option_idx * (n + THREADS_PER_BLOCK) + threadId;

    // entry i stores value corresponding to exponent 2*i - n
    layer_values[layer_index] = st_buffer_bank0[st_index] = fmax(
        sign[option_idx] * fma(S[option_idx], u_pow_2_threadId * u_pow_minus_n, -K[option_idx]),
        0.0);

    // entry i stores value corresponding to exponent 2*i - n + 1
    st_buffer_bank1[st_index] =
        fmax(sign[option_idx] * fma(S[option_idx], u_pow_2_threadId * u_pow_minus_n * u[option_idx],
                                    -K[option_idx]),
             0.0);
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel_batch)(
    const double* __restrict__ layer_values_read, double* __restrict__ layer_values_write,
    const double* __restrict__ st_buffer_bank0, const double* __restrict__ st_buffer_bank1,
    const double* __restrict__ up, const double* __restrict__ down, const int level,
    const int n, const int* __restrict__ upper_bound) {
    const int option_idx = blockIdx.y;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * option_idx;

    __shared__ double warp_edges_layer_values_tile[NUM_WARPS + 1];

    const int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    const int tile_base = tile_stride * blockIdx.x;
    const int node_id = tile_base + threadIdx.x;
    if (node_id > upper_bound[option_idx]) return;

    double val = layer_values_read[base_layer_values + node_id];
    __syncwarp();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const double* st_buffer_bank =
            (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        double up_val = __shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];

        double hold = fma(up[option_idx], up_val, down[option_idx] * val);
        int st_index = node_id + (n - current_level) / 2;
        double exercise = st_buffer_bank[base_st_buffer + st_index];
        val = fmax(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_layer_values + node_id] = val;
    }
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

    __shared__ double warp_edges_layer_values_tile[NUM_WARPS + 1];

    int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    int tile_base = tile_stride * blockIdx.x;
    int node_id = tile_base + threadIdx.x;

    double val = layer_values_read[node_id];
    __syncwarp();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;
        const double* st_buffer_bank = (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        double up_val = __shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];

        double hold = fma(up, up_val, down * val);
        int st_index = node_id + (n - current_level) / 2;
        double exercise = st_buffer_bank[st_index];
        val = fmax(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[node_id] = val;
    }
}

/*
    At each layer l exercise value of node i (from the bottom) is calculated with the following
    exponent: 2*i - l = 2 * (i + (n - l) / 2) - n           if (n - l) even
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

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layer_kernel_batch)(
    const double* __restrict__ layer_values_read, double* __restrict__ layer_values_write,
    const double* __restrict__ st_buffer_bank0, const double* __restrict__ st_buffer_bank1,
    const double* __restrict__ up, const double* __restrict__ down, const int level,
    const int n) {
    const int option_idx = blockIdx.y;
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;
    const int base_st_buffer = (n + THREADS_PER_BLOCK + UNROLL_FACTOR) * option_idx;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double hold = up[option_idx] * layer_values_read[base_layer_values + threadId + 1] +
                  down[option_idx] * layer_values_read[base_layer_values + threadId];
    const double* st_buffer_bank =
        (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
    double exercise =
        st_buffer_bank[base_st_buffer + threadId + (n - level) / 2];
    layer_values_write[base_layer_values + threadId] = fmax(hold, exercise);
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

template <const Hyperparams& h>
void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out) {
    /// all n are equal
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    std::vector<double> h_up(num_runs), h_down(num_runs);
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

        int bound = FUNC_NAME(search_bound)(run.n,run.S,run.K, u,option_type_sign(run.type)); 
        h_bound[i] = bound;
        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_up[i] = p * risk_free_rate;
        h_down[i] = one_minus_p * risk_free_rate;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);
    }

    double *d_S, *d_K, *d_u, *d_up, *d_down;
    int *d_bound;
    int *d_n_arr, *d_sign;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_u, num_runs * sizeof(double));
    cudaMalloc(&d_up, num_runs * sizeof(double));
    cudaMalloc(&d_down, num_runs * sizeof(double));
    cudaMalloc(&d_n_arr, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_bound, num_runs * sizeof(int));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_down.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_arr, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bound, h_bound.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    int n = runs[0].n;
    const int layer_size = num_runs* (n + THREADS_PER_BLOCK);
    double *layer_values_read_d, *layer_values_write_d;
    cudaMalloc(&layer_values_read_d, layer_size * sizeof(double));
    cudaMalloc(&layer_values_write_d, layer_size * sizeof(double));

    const int buffer_size = num_runs * (n + THREADS_PER_BLOCK + UNROLL_FACTOR);
    double *st_buffer_bank0_d, *st_buffer_bank1_d;
    cudaMalloc(&st_buffer_bank0_d, buffer_size * sizeof(double));
    cudaMalloc(&st_buffer_bank1_d, buffer_size * sizeof(double));

    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);
    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR><<<num_blocks_2d, THREADS_PER_BLOCK>>>(
        st_buffer_bank0_d, st_buffer_bank1_d, d_S, d_K, d_u, d_sign, n, layer_values_read_d);
    int level = n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));

        dim3 num_blocks_2d(num_blocks, num_runs);
        FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks_2d, THREADS_PER_BLOCK>>>(layer_values_read_d, layer_values_write_d,
                                                   st_buffer_bank0_d, st_buffer_bank1_d, d_up, d_down,
                                                   level, n, d_bound);
        std::swap(layer_values_read_d, layer_values_write_d);
    }
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        num_blocks = std::ceil((level + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK));
        dim3 num_blocks_2d(num_blocks, num_runs);
        FUNC_NAME(compute_next_layer_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR><<<num_blocks_2d, THREADS_PER_BLOCK>>>(
            layer_values_read_d, layer_values_write_d, st_buffer_bank0_d, st_buffer_bank1_d, d_up,
            d_down, level, n);
        std::swap(layer_values_read_d, layer_values_write_d);
    }

    cudaDeviceSynchronize();
    std::vector<double> h_results_store(layer_size);
    cudaMemcpy(h_results_store.data(), layer_values_read_d, layer_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_runs; ++i) {
        out[i] = h_results_store[i * (n + THREADS_PER_BLOCK)];
    }

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(layer_values_read_d);
    cudaFree(layer_values_write_d);
    cudaFree(st_buffer_bank0_d);
    cudaFree(st_buffer_bank1_d);
    checkCuda(cudaGetLastError());
}

template double FUNC_NAME(
    vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template void FUNC_NAME(vanilla_american_binomial_cuda_batch)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(
    std::vector<PricingInput>& runs, std::vector<double>& out);

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
