#pragma once

#include "constants.hpp"

double vanilla_american_binomial_cpu_naive(const double S, const double K, const double T,
                                           const double r, const double sigma, const double q,
                                           const int n, const OptionType type);

double vanilla_american_binomial_cpu_trimotm(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type);

double vanilla_american_binomial_cpu_trimotm_stprcmp(const double S, const double K, const double T,
                                                     const double r, const double sigma,
                                                     const double q, const int n,
                                                     const OptionType type);

double vanilla_american_binomial_cpu_trimotm_trimeeoff_stprcmp(const double S, const double K,
                                                               const double T, const double r,
                                                               const double sigma, const double q,
                                                               const int n, const OptionType type);

double vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp(const double S, const double K,
                                                              const double T, const double r,
                                                              const double sigma, const double q,
                                                              const int n, const OptionType type);

inline double vanilla_american_binomial_cpu(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type) {
    // Choose the current best backend here:
    return vanilla_american_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
}
