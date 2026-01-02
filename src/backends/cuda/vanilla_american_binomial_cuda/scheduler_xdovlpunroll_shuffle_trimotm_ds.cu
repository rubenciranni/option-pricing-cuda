
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>
#include <map>

#include "backends/cuda/ds_float.cuh"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "constants.hpp"

#define IMPL_NAME scheduler_xdovlpunroll_shuffle_trimotm_ds
#define WARP_SIZE 32

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

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(fill_st_buffers_kernel_batch)(const double* __restrict__ S,
                                                        const double* __restrict__ K,
                                                        const double* __restrict__ u,
                                                        const int* __restrict__ sign, const int n,
                                                        ds_float* __restrict__ layer_values) {
    const int option_idx = blockIdx.y;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculation logic uses double for the power steps
    const double u_pow_2_threadId = pow(u[option_idx], (double)2 * threadId);
    const double u_pow_minus_n = pow(u[option_idx], (double)-n);

    const int layer_index = option_idx * (n + THREADS_PER_BLOCK) + threadId;

    // entry i stores value corresponding to exponent 2*i - n
    // Compute in double -> Convert to DS -> Store
    ds_float val0 = double_to_ds(fmax(
        sign[option_idx] * fma(S[option_idx], u_pow_2_threadId * u_pow_minus_n, -K[option_idx]),
        0.0));
    layer_values[layer_index] = val0;
}

template <const int THREADS_PER_BLOCK, const int UNROLL_FACTOR>
__global__ void FUNC_NAME(compute_next_layers_kernel_batch)(
    const ds_float* __restrict__ layer_values_read, ds_float* __restrict__ layer_values_write,
    const ds_float* __restrict__ up, const ds_float* __restrict__ down, const int level,
    const int n, const int* __restrict__ upper_bound, const int MAX_UNROLL_FACTOR,
    const double* __restrict__ S, const double* __restrict__ K, const double* __restrict__ u,
    const int* __restrict__ sign) {
    const int option_idx = blockIdx.y;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    const unsigned int full_mask = 0xffffffff;
    const unsigned int active_mask = full_mask & ~(1 << (WARP_SIZE - 1));
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    // Offsets
    const int base_layer_values = (n + THREADS_PER_BLOCK) * option_idx;

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

    // Precompute values for exercise calculation
    const double S_opt = S[option_idx];
    const double K_opt = K[option_idx];
    const double u_opt = u[option_idx];
    const int sign_opt = sign[option_idx];

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int current_level = level + UNROLL_FACTOR - 1 - i;

        // DS Shuffle
        ds_float up_val = ds_shfl_down_sync(active_mask, val, 1);

        if (lane_id == 0) warp_edges_layer_values_tile[warp_id] = val;
        __syncthreads();
        if (lane_id == WARP_SIZE - 1) up_val = warp_edges_layer_values_tile[warp_id + 1];
        __syncthreads();

        // DS Math: hold = up * up_val + down * val
        ds_float hold = ds_add_two_mults_streamlined(my_up, up_val, my_down, val);

        int exp_idx = 2 * node_id - current_level;  // exponent for u term at this node
        double exercise_double =
            fmax(sign_opt * (S_opt * pow(u_opt, (double)exp_idx) - K_opt), 0.0);
        ds_float exercise = double_to_ds(exercise_double);

        val = ds_max(hold, exercise);
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_layer_values + node_id] = val;
    }
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

void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
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

    // 1. Fill buffers
    int num_blocks = std::ceil((n + 1) * 1.0 / THREADS_PER_BLOCK);
    dim3 num_blocks_2d(num_blocks, num_runs);
    FUNC_NAME(fill_st_buffers_kernel_batch)<THREADS_PER_BLOCK, MAX_UNROLL_FACTOR>
        <<<num_blocks_2d, THREADS_PER_BLOCK>>>(d_S, d_K, d_u, d_sign, n, layer_values_read_d);

    int level = n;

    auto get_current_unroll_factor = [](int current_level) -> int {
        if (current_level >= (1 << 20)) {
            return 16;
        } else if (current_level >= (1 << 14)) {
            return 24;
        } else if (current_level >= (1 << 11)) {
            return 32;
        } else if (current_level >= 512) {
            return 64;
        } else if (current_level >= (32)) {
            return 32;
        } else if (current_level >= (16)) {
            return 16;
        } else if (current_level >= (8)) {
            return current_level;
        } else {
            return 1;
        }
    };

    // Launch the correct templated kernel based on the chosen unroll factor.
    for (; level > 0;) {
        int U = get_current_unroll_factor(level);

        // if (lastU!= U) {
        //     nvtxRangePushA(kernel_label.c_str());
        //     cudaProfilerStart();
        // }
        num_blocks = std::ceil((level)*1.0 / (THREADS_PER_BLOCK - U));
        dim3 num_blocks_2d_loop(num_blocks, num_runs);

           
#define CASE_N(N) \
    case N: \
        FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, N> \
            <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>( \
                layer_values_read_d, layer_values_write_d,  \
                d_up, d_down, level - U, n, d_bound, MAX_UNROLL_FACTOR, \
                d_S, d_K, d_u, d_sign); \
        break;

        switch (U) {
            CASE_N(1)
            CASE_N(2)
            CASE_N(3)
            CASE_N(4)
            CASE_N(5)
            CASE_N(6)
            CASE_N(7)
            CASE_N(8)
            CASE_N(16)
            CASE_N(24)
            CASE_N(32)
            CASE_N(64)
            CASE_N(128)

            default:
                // Fallback to U=1 if an unsupported unroll factor is requested.
                FUNC_NAME(compute_next_layers_kernel_batch)<THREADS_PER_BLOCK, 1>
                    <<<num_blocks_2d_loop, THREADS_PER_BLOCK>>>(
                        layer_values_read_d, layer_values_write_d, d_up, d_down, level - 1, n,
                        d_bound, MAX_UNROLL_FACTOR, d_S, d_K, d_u, d_sign);
                U = 1;
                break;
        }

        std::swap(layer_values_read_d, layer_values_write_d);
        level -= U;
        // if (lastU!= U) {
        //     cudaDeviceSynchronize();
        //     cudaProfilerStop();
        //     nvtxRangePop();
        //     lastU = U;
        // }
    }

    num_blocks = std::ceil(num_runs * 1.0 / THREADS_PER_BLOCK);
    FUNC_NAME(copy_final_value)<THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(layer_values_read_d, d_out, n, num_runs);

    cudaMemcpy(out.data(), d_out, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFreeAsync(d_S, 0);
    cudaFreeAsync(d_K, 0);
    cudaFreeAsync(d_u, 0);
    cudaFreeAsync(d_up, 0);
    cudaFreeAsync(d_down, 0);
    cudaFreeAsync(d_n_arr, 0);
    cudaFreeAsync(d_sign, 0);
    cudaFreeAsync(d_bound, 0);
    cudaFreeAsync(d_out, 0);
    cudaFreeAsync(layer_values_read_d, 0);
    cudaFreeAsync(layer_values_write_d, 0);
    checkCuda(cudaGetLastError());
}
