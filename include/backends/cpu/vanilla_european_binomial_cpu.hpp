#pragma once

#include "constants.hpp"

double vanilla_european_binomial_cpu_naive(const double S, const double K, const double T,
                                           const double r, const double sigma, const double q,
                                           const int n, const OptionType type);

inline double vanilla_european_binomial_cpu(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type) {
    // Choose the current best backend here:
    return vanilla_european_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
}
