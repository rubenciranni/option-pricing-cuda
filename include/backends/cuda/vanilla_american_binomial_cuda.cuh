#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <vector>

#include "backends/hyperparams.hpp"
#include "constants.hpp"

// Include modular header files for organization
#include "vanilla_american_binomial_cuda_basic.cuh"
#include "vanilla_american_binomial_cuda_templated.cuh"
#include "vanilla_american_binomial_cuda_batch.cuh"

#define CONCAT_IMPL(a, b) a##_##b
#define EXPAND_AND_CONCAT(a, b) CONCAT_IMPL(a, b)
#define FUNC_NAME(func) EXPAND_AND_CONCAT(func, IMPL_NAME)

inline double vanilla_american_binomial_cuda(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type) {
    // Choose the current best backend here:
    return vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm<
        DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>(S, K, T, r, sigma, q, n, type);
}

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

// Declaration only; definition is in check_occupancy.cu to avoid multiple definitions
std::pair<int, int> check_occupancy_function(CUfunction func);