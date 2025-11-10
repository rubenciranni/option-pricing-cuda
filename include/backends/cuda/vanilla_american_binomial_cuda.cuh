#pragma once

#include "backends/hyperparams.hpp"
#include "constants.hpp"

double vanilla_american_binomial_cuda_naive(const double S, const double K, const double T,
                                            const double r, const double sigma, const double q,
                                            const int n, const OptionType type);

double vanilla_american_binomial_cuda_no_sync(const double S, const double K, const double T,
                                              const double r, const double sigma, const double q,
                                              const int n, const OptionType type);

double vanilla_american_binomial_cuda_fill(const double S, const double K, const double T,
                                           const double r, const double sigma, const double q,
                                           const int n, const OptionType type);

double vanilla_american_binomial_cuda_fill_banked(const double S, const double K, const double T,
                                                  const double r, const double sigma,
                                                  const double q, const int n,
                                                  const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_tile(const double S, const double K, const double T,
                                           const double r, const double sigma, const double q,
                                           const int n, const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_unroll(const double S, const double K, const double T,
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

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_unroll_tile(const double S, const double K, const double T,
                                                  const double r, const double sigma,
                                                  const double q, const int n,
                                                  const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_x_y_unroll_new(const double S, const double K, const double T,
                                                     const double r, const double sigma,
                                                     const double q, const int n,
                                                     const OptionType type);

template <const Hyperparams& h>
double vanilla_american_binomial_cuda_x_y_unroll(const double S, const double K, const double T,
                                                 const double r, const double sigma, const double q,
                                                 const int n, const OptionType type);

double vanilla_american_binomial_cuda_x_y_unroll_tile(const double S, const double K,
                                                      const double T, const double r,
                                                      const double sigma, const double q,
                                                      const int n, const OptionType type);

double vanilla_american_binomial_cuda_overlap_unroll(const double S, const double K, const double T,
                                                     const double r, const double sigma,
                                                     const double q, const int n,
                                                     const OptionType type);

double vanilla_american_binomial_cuda_mem(const double S, const double K, const double T,
                                          const double r, const double sigma, const double q,
                                          const int n, const OptionType type);

double vanilla_american_binomial_cuda_x_y_unroll_tile_banked_ignore(const double S, const double K,
                                                                    const double T, const double r,
                                                                    const double sigma,
                                                                    const double q, const int n,
                                                                    const OptionType type);

inline double vanilla_american_binomial_cuda(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type) {
    // Choose the current best backend here:
    return vanilla_american_binomial_cuda_overlap_unroll(S, K, T, r, sigma, q, n, type);
}
