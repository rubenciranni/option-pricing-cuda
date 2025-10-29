#pragma once

#include "constants.hpp"

double vanilla_american_binomial_cuda_naive(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type);

double vanilla_american_binomial_cuda_no_sync(const double S, const double K, const double T,
                                              const double r, const double sigma, const double q,
                                              const int n, const OptionType type);

double vanilla_american_binomial_cuda_precomputed_stock_price(const double S, const double K,
                                                               const double T, const double r,
                                                               const double sigma, const double q,
                                                               const int n, const OptionType type);

double vanilla_american_binomial_cuda_precomputed_payoffs(const double S, const double K,
                                                           const double T, const double r,
                                                           const double sigma, const double q,
                                                           const int n, const OptionType type);

inline double vanilla_american_binomial_cuda(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type) {
  // Choose the current best backend here:
  return vanilla_american_binomial_cuda_precomputed_payoffs(S, K, T, r, sigma, q, n, type);
}
