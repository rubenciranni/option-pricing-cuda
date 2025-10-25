#pragma once

#include "constants.hpp"

double vanilla_american_binomial_openmp_naive(const double S, const double K, const double T,
                                              const double r, const double sigma, const double q,
                                              const int n, const OptionType type);

inline double vanilla_american_binomial_openmp(const double S, const double K, const double T,
                                               const double r, const double sigma, const double q,
                                               const int n, const OptionType type) {
  // Choose the current best backend here:
  return vanilla_american_binomial_openmp_naive(S, K, T, r, sigma, q, n, type);
}
