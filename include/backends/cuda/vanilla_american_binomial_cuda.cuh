#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "backends/hyperparams.hpp"
#include "constants.hpp"

#define CONCAT_IMPL(a, b) a##_##b
#define EXPAND_AND_CONCAT(a, b) CONCAT_IMPL(a, b)
#define FUNC_NAME(func) EXPAND_AND_CONCAT(func, IMPL_NAME)

double vanilla_american_binomial_cuda_naive(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp(const double S, const double K, const double T,
                                              const double r, const double sigma, const double q,
                                              const int n, const OptionType type);

double vanilla_american_binomial_cuda_bkdstprcmp(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_yunroll_vtile(const double S, const double K,
                                                            const double T, const double r,
                                                            const double sigma, const double q,
                                                            const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_xunroll_vprftc(const double S, const double K,
                                                             const double T, const double r,
                                                             const double sigma, const double q,
                                                             const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_xunroll_stvtile(const double S, const double K,
                                                              const double T, const double r,
                                                              const double sigma, const double q,
                                                              const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_xyunroll_vprftc(const double S, const double K,
                                                              const double T, const double r,
                                                              const double sigma, const double q,
                                                              const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvprftc(const double S, const double K,
                                                                const double T, const double r,
                                                                const double sigma, const double q,
                                                                const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc_trimotm(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile(const double S, const double K,
                                                                    const double T, const double r,
                                                                    const double sigma,
                                                                    const double q, const int n,
                                                                    const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xovlpunroll_vtile(const double S, const double K,
                                                                   const double T, const double r,
                                                                   const double sigma,
                                                                   const double q, const int n,
                                                                   const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_malloc(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

inline double vanilla_american_binomial_cuda(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type) {
    // Choose the current best backend here:
    return vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm<
        DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000>(S, K, T, r, sigma, q, n, type);
}

std::vector<double> vanilla_american_binomial_cuda_batch_naive(std::vector<PricingInput> runs);

std::vector<double> vanilla_american_binomial_cuda_batch_stprcmp(std::vector<PricingInput> runs);

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}
