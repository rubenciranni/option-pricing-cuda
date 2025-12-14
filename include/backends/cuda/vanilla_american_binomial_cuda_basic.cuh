#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.hpp"

// Basic (non-templated) CUDA kernel declarations
// These are simple implementations without template parameters

double vanilla_american_binomial_cuda_naive(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type);

double vanilla_american_binomial_cuda_nvidia_baseline(const double S, const double K,
                                                      const double T, const double r,
                                                      const double sigma, const double q,
                                                      const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp(const double S, const double K, const double T,
                                              const double r, const double sigma, const double q,
                                              const int n, const OptionType type);

double vanilla_american_binomial_cuda_bkdstprcmp(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

double vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc_trimotm(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);

double vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile(const double S, const double K,
                                                                    const double T, const double r,
                                                                    const double sigma,
                                                                    const double q, const int n,
                                                                    const OptionType type);
