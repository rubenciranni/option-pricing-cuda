#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "backends/hyperparams.hpp"
#include "constants.hpp"

// Templated CUDA kernel declarations
// These kernels are parametrized by Hyperparams and compiled for different configurations

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

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvprftc(const double S, const double K,
                                                                const double T, const double r,
                                                                const double sigma, const double q,
                                                                const int n, const OptionType type);

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

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_malloc(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_float(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);
