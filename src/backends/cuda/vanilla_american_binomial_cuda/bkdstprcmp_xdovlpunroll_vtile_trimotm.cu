#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

// TPB 128 UF 35 ~2.6ms on 10k
// TPB 256 UF 32 ~660ms on 250k

#define IMPL_NAME bkdstprcmp_xdovlpunroll_vtile_trimotm

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
    double* __restrict__ layer_values_read, double* __restrict__ layer_values_write,
    double* __restrict__ st_buffer_bank0, double* __restrict__ st_buffer_bank1, const double up,
    const double down, const int level, const int n, const int upper_bound) {
    __shared__ double layer_values_tile[2][THREADS_PER_BLOCK + 1];

    int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    int tile_base = tile_stride * blockIdx.x;
    int node_id = tile_base + threadIdx.x;

    layer_values_tile[0][threadIdx.x] = layer_values_read[node_id];

    __syncthreads();

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int read_idx = i % 2;
        int write_idx = (i + 1) % 2;

        int current_level = level + UNROLL_FACTOR - 1 - i;
        double* st_buffer_bank = (n - current_level) % 2 ? st_buffer_bank1 : st_buffer_bank0;

        double hold = fma(up, layer_values_tile[read_idx][threadIdx.x + 1],
                          down * layer_values_tile[read_idx][threadIdx.x]);
        int st_index = node_id + (n - current_level) / 2;
        double exercise = st_buffer_bank[st_index];
        layer_values_tile[write_idx][threadIdx.x] = fmax(hold, exercise);

        __syncthreads();
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[node_id] = layer_values_tile[UNROLL_FACTOR % 2][threadIdx.x];
    }
}

/*
    At each layer l exercise value of node i (from the bottom) is calculated with the following
    exponent: 2*i - l = 2 * (i + (n - l) / 2) - n           if (n - l) even
    the correspoding value is stored at st_buffer_bank0[i + (n - l) / 2]

    2*i - l = 2 * (i + (n - l - 1) / 2) - n + 1   if (n - l) odd
    the correspoding value is stored at st_buffer_bank1[i + (n - l) / 2]
*/
__global__ void FUNC_NAME(compute_next_layer_kernel)(double* __restrict__ layer_values_read,
                                                     double* __restrict__ layer_values_write,
                                                     double* __restrict__ st_buffer_bank0,
                                                     double* __restrict__ st_buffer_bank1,
                                                     const double up, const double down,
                                                     const int level, const int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double hold = up * layer_values_read[threadId + 1] + down * layer_values_read[threadId];
    double* st_buffer_bank = (n - level) % 2 ? st_buffer_bank1 : st_buffer_bank0;
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
    const double delta_t = T / n;
    const double u = std::exp(sigma * std::sqrt(delta_t));
    const double d = 1.0 / u;
    const double p = (exp((r - q) * delta_t) - d) / (u - d);
    const double discount = std::exp(-r * delta_t);
    const double up = p * discount;
    const double down = (1.0 - p) * discount;
    const int sign = option_type_sign(type);

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

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

    checkCuda(cudaGetLastError());
    return value_h;
}

// Batch processing kernels
__global__ void FUNC_NAME(fill_st_buffers_kernel_batch)(
    double* __restrict__ d_st_buffer_bank0, double* __restrict__ d_st_buffer_bank1,
    const double* d_S, const double* d_K, const double* d_u, const int* d_sign, const int* d_n,
    const int max_n) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (threadId > n) return;

    double S = d_S[option_idx];
    double K = d_K[option_idx];
    double u = d_u[option_idx];
    int sign = d_sign[option_idx];

    double u_pow_2_threadId = pow(u, (double)2 * threadId);
    double u_pow_minus_n = pow(u, (double)-n);

    int base_idx = option_idx * (max_n + 1);
    // entry i stores value corresponding to exponent 2*i - n
    d_st_buffer_bank0[base_idx + threadId] =
        fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n, -K), 0.0);

    // entry i stores value corresponding to exponent 2*i - n + 1
    d_st_buffer_bank1[base_idx + threadId] =
        fmax(sign * fma(S, u_pow_2_threadId * u_pow_minus_n * u, -K), 0.0);
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel_batch)(
    double* __restrict__ d_layer_values_read, double* __restrict__ d_layer_values_write,
    double* __restrict__ d_st_buffer_bank0, double* __restrict__ d_st_buffer_bank1,
    const double* d_up, const double* d_down, const int* d_n, const int* d_bound, const int max_n,
    const int level) {
    int option_idx = blockIdx.y;
    int n = d_n[option_idx];
    int bound = d_bound[option_idx];

    if (level >= n) return;

    __shared__ double layer_values_tile[2][THREADS_PER_BLOCK + 1];

    int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    int tile_base = tile_stride * blockIdx.x;
    int node_id = tile_base + threadIdx.x;
    int base_idx = option_idx * (max_n + 1);

    if (node_id <= level)
        layer_values_tile[0][threadIdx.x] = d_layer_values_read[base_idx + node_id];

    __syncthreads();

    double up = d_up[option_idx];
    double down = d_down[option_idx];

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int read_idx = i % 2;
        int write_idx = (i + 1) % 2;

        int current_level = level + UNROLL_FACTOR - 1 - i;
        if (current_level < 0 || node_id > current_level || node_id > bound) break;

        double* st_buffer_bank = (n - current_level) % 2 ? d_st_buffer_bank1 : d_st_buffer_bank0;

        double hold = fma(up, layer_values_tile[read_idx][threadIdx.x + 1],
                          down * layer_values_tile[read_idx][threadIdx.x]);
        int st_index = base_idx + node_id + (n - current_level) / 2;
        double exercise = st_buffer_bank[st_index];
        layer_values_tile[write_idx][threadIdx.x] = fmax(hold, exercise);

        __syncthreads();
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR && node_id < level + 1 &&
        node_id <= bound) {
        d_layer_values_write[base_idx + node_id] =
            layer_values_tile[UNROLL_FACTOR % 2][threadIdx.x];
    }
}

__global__ void FUNC_NAME(compute_next_layer_kernel_batch)(
    double* __restrict__ d_layer_values_read, double* __restrict__ d_layer_values_write,
    double* __restrict__ d_st_buffer_bank0, double* __restrict__ d_st_buffer_bank1,
    const double* d_up, const double* d_down, const int* d_n, const int max_n, const int level) {
    int option_idx = blockIdx.y;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_n[option_idx];

    if (level >= n || threadId > level) return;

    int base_idx = option_idx * (max_n + 1);

    double up = d_up[option_idx];
    double down = d_down[option_idx];

    double hold = up * d_layer_values_read[base_idx + threadId + 1] +
                  down * d_layer_values_read[base_idx + threadId];
    double* st_buffer_bank = (n - level) % 2 ? d_st_buffer_bank1 : d_st_buffer_bank0;
    double exercise = st_buffer_bank[base_idx + threadId + (n - level) / 2];
    d_layer_values_write[base_idx + threadId] = fmax(hold, exercise);
}

template <const Hyperparams& h>
void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    constexpr int THREADS_PER_BLOCK = h.THREADS_PER_BLOCK;
    constexpr int UNROLL_FACTOR = h.UNROLL_FACTOR;

    int max_n = 0;
    std::vector<double> h_S(num_runs), h_K(num_runs), h_u(num_runs);
    std::vector<double> h_up(num_runs), h_down(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs), h_bound(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];
        if (run.n > max_n) max_n = run.n;

        const double deltaT = run.T / run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        const double one_minus_p = 1.0 - p;
        const int sign = option_type_sign(run.type);

        h_S[i] = run.S;
        h_K[i] = run.K;
        h_u[i] = u;
        h_up[i] = p * risk_free_rate;
        h_down[i] = one_minus_p * risk_free_rate;
        h_n[i] = run.n;
        h_sign[i] = sign;
        h_bound[i] = FUNC_NAME(search_bound)(run.n, run.S, run.K, u, sign);
    }

    double *d_S, *d_K, *d_u, *d_up, *d_down;
    int *d_n_arr, *d_sign, *d_bound;

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

    double *d_layer_values_read, *d_layer_values_write;
    size_t layer_size = num_runs * (max_n + 1);
    cudaMalloc(&d_layer_values_read, layer_size * sizeof(double));
    cudaMalloc(&d_layer_values_write, layer_size * sizeof(double));

    double *d_st_buffer_bank0, *d_st_buffer_bank1;
    cudaMalloc(&d_st_buffer_bank0, layer_size * sizeof(double));
    cudaMalloc(&d_st_buffer_bank1, layer_size * sizeof(double));

    int fill_blocks_x = std::ceil((max_n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 fill_blocks(fill_blocks_x, num_runs);

    FUNC_NAME(fill_st_buffers_kernel_batch)<<<fill_blocks, THREADS_PER_BLOCK>>>(
        d_st_buffer_bank0, d_st_buffer_bank1, d_S, d_K, d_u, d_sign, d_n_arr, max_n);

    // Copy first layer (layer n) from st_buffer_bank0
    cudaMemcpy(d_layer_values_read, d_st_buffer_bank0, layer_size * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // Compute layers with unrolling
    int level = max_n - 1 - (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= UNROLL_FACTOR) {
        int num_blocks_x =
            std::ceil((max_n + UNROLL_FACTOR) * 1.0 / (THREADS_PER_BLOCK - UNROLL_FACTOR));
        dim3 num_blocks(num_blocks_x, num_runs);
        FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, UNROLL_FACTOR>
            <<<num_blocks, THREADS_PER_BLOCK>>>(d_layer_values_read, d_layer_values_write,
                                                d_st_buffer_bank0, d_st_buffer_bank1, d_up, d_down,
                                                d_n_arr, d_bound, max_n, level);
        std::swap(d_layer_values_read, d_layer_values_write);
    }

    // Remaining layers without unrolling
    level += (UNROLL_FACTOR - 1);
    for (; level >= 0; level -= 1) {
        int num_blocks_x = std::ceil((level + 1) * 1.0 / THREADS_PER_BLOCK);
        dim3 num_blocks(num_blocks_x, num_runs);
        FUNC_NAME(compute_next_layer_kernel_batch)<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_layer_values_read, d_layer_values_write, d_st_buffer_bank0, d_st_buffer_bank1, d_up,
            d_down, d_n_arr, max_n, level);
        std::swap(d_layer_values_read, d_layer_values_write);
    }

    cudaDeviceSynchronize();

    std::vector<double> h_results_store(layer_size);
    cudaMemcpy(h_results_store.data(), d_layer_values_read, layer_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_runs; ++i) {
        out[i] = h_results_store[i * (max_n + 1)];
    }

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_u);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_n_arr);
    cudaFree(d_sign);
    cudaFree(d_bound);
    cudaFree(d_layer_values_read);
    cudaFree(d_layer_values_write);
    cudaFree(d_st_buffer_bank0);
    cudaFree(d_st_buffer_bank1);
    checkCuda(cudaGetLastError());
}

template double FUNC_NAME(
    vanilla_american_binomial_cuda)<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template void FUNC_NAME(vanilla_american_binomial_cuda_batch)<
    DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000>(std::vector<PricingInput>& runs,
                                                                 std::vector<double>& out);

#ifdef DO_CARTESIAN_PRODUCT
#ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM

#define PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM(    \
    ID, A, B, C, D, E, Y)                                                                    \
    template double FUNC_NAME(vanilla_american_binomial_cuda)<GRID_SEARCH_HYPERPARAMS_##ID>( \
        const double S, const double K, const double T, const double r, const double sigma,  \
        const double q, const int n, const OptionType type);
APPLY_FUNCTION(PRODUCE_INSTANCES_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM,
               HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif
