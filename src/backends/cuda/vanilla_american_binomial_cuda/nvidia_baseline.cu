/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "constants.hpp"

#define IMPL_NAME nvidia_baseline
#define THREADS_PER_BLOCK 1024

namespace cg = cooperative_groups;

__device__ inline double expiryValue(double S, double X, double vDt, int i, int n, int sign) {
    double d = S * exp(vDt * (2 * i - n));
    double result = sign * (d - X);
    return (result > 0.0) ? result : 0.0;
}

__global__ void FUNC_NAME(binomial_options_kernel)(double* d_CallValue, const double* d_S,
                                                   const double* d_K, const double* d_vDt,
                                                   const double* d_puByDf, const double* d_pdByDf,
                                                   const int* d_n, const int* d_sign) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ double call_exchange[THREADS_PER_BLOCK + 1];

    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;

    const double S = d_S[block_idx];
    const double X = d_K[block_idx];
    const double vDt = d_vDt[block_idx];
    const double puByDf = d_puByDf[block_idx];
    const double pdByDf = d_pdByDf[block_idx];
    const int n = d_n[block_idx];
    const int sign = d_sign[block_idx];

    const int elems_per_thread = n / THREADS_PER_BLOCK;

    double call[1024 + 1];

#pragma unroll
    for (int i = 0; i < elems_per_thread; ++i)
        call[i] = expiryValue(S, X, vDt, tid * elems_per_thread + i, n, sign);

    if (tid == 0) call_exchange[THREADS_PER_BLOCK] = expiryValue(S, X, vDt, n, n, sign);

    int final_it = max(0, tid * elems_per_thread - 1);

#pragma unroll 16
    for (int i = n; i > 0; --i) {
        call_exchange[tid] = call[0];
        cg::sync(cta);
        call[elems_per_thread] = call_exchange[tid + 1];
        cg::sync(cta);

        if (i > final_it) {
#pragma unroll
            for (int j = 0; j < elems_per_thread; ++j) {
                double continuation_value = puByDf * call[j + 1] + pdByDf * call[j];
                double fwd = S * exp(vDt * (2 * (tid * elems_per_thread + j) - i + 1));
                double exercise_value = (sign * (fwd - X) > 0.0) ? sign * (fwd - X) : 0.0;
                call[j] =
                    (exercise_value > continuation_value) ? exercise_value : continuation_value;
            }
        }
    }

    if (tid == 0) {
        d_CallValue[block_idx] = call[0];
    }
}

double FUNC_NAME(vanilla_american_binomial_cuda)(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type) {
    std::vector<PricingInput> runs(1);
    runs[0].S = S;
    runs[0].K = K;
    runs[0].T = T;
    runs[0].r = r;
    runs[0].sigma = sigma;
    runs[0].q = q;
    runs[0].n = n;
    runs[0].type = type;

    std::vector<double> results(1);
    FUNC_NAME(vanilla_american_binomial_cuda_batch)(runs, results);
    return results[0];
}

void FUNC_NAME(vanilla_american_binomial_cuda_batch)(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out) {
    size_t num_runs = runs.size();
    if (num_runs == 0) return;

    std::vector<double> h_S(num_runs), h_K(num_runs);
    std::vector<double> h_vDt(num_runs), h_puByDf(num_runs), h_pdByDf(num_runs);
    std::vector<int> h_n(num_runs), h_sign(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        const PricingInput& run = runs[i];
        const double deltaT = run.T / (double)run.n;
        const double u = std::exp(run.sigma * std::sqrt(deltaT));
        const double d = 1.0 / u;
        const double p = (exp((run.r - run.q) * deltaT) - d) / (u - d);
        const double risk_free_rate = std::exp(-run.r * deltaT);
        if (run.n > 1024) {
            std::cerr
                << "Error: n exceeds maximum supported steps (1024) for NVIDIA baseline kernel."
                << std::endl;
            return;
        }
        if (run.n % (THREADS_PER_BLOCK) != 0) {
            std::cerr << "Error: n must be a multiple of " << THREADS_PER_BLOCK
                      << " for NVIDIA baseline kernel." << std::endl;
            return;
        }

        h_S[i] = run.S;
        h_K[i] = run.K;
        h_vDt[i] = run.sigma * std::sqrt(deltaT);
        h_puByDf[i] = p * risk_free_rate;
        h_pdByDf[i] = (1.0 - p) * risk_free_rate;
        h_n[i] = run.n;
        h_sign[i] = option_type_sign(run.type);
    }

    double *d_S, *d_K, *d_vDt, *d_puByDf, *d_pdByDf, *d_CallValue;
    int *d_n, *d_sign;

    cudaMalloc(&d_S, num_runs * sizeof(double));
    cudaMalloc(&d_K, num_runs * sizeof(double));
    cudaMalloc(&d_vDt, num_runs * sizeof(double));
    cudaMalloc(&d_puByDf, num_runs * sizeof(double));
    cudaMalloc(&d_pdByDf, num_runs * sizeof(double));
    cudaMalloc(&d_n, num_runs * sizeof(int));
    cudaMalloc(&d_sign, num_runs * sizeof(int));
    cudaMalloc(&d_CallValue, num_runs * sizeof(double));

    cudaMemcpy(d_S, h_S.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vDt, h_vDt.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_puByDf, h_puByDf.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pdByDf, h_pdByDf.data(), num_runs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sign, h_sign.data(), num_runs * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(num_runs);

    FUNC_NAME(binomial_options_kernel)<<<grid_dim, block_dim>>>(d_CallValue, d_S, d_K, d_vDt,
                                                                d_puByDf, d_pdByDf, d_n, d_sign);

    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_CallValue, num_runs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_vDt);
    cudaFree(d_puByDf);
    cudaFree(d_pdByDf);
    cudaFree(d_n);
    cudaFree(d_sign);
    cudaFree(d_CallValue);

    checkCuda(cudaGetLastError());
}
