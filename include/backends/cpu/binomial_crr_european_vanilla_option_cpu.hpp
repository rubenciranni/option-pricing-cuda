#pragma once
#include "backends/cpu/binomial_crr_european_vanilla_option_cpu.hpp"
#include "constants.hpp"

/**
 * @brief Prices an American option using the Cox–Ross–Rubinstein (CRR) binomial model.
 *
 * @param S      Current underlying asset price.
 * @param K      Strike price of the option.
 * @param T      Time to maturity (in years).
 * @param r      Risk-free interest rate (annualized).
 * @param sigma  Volatility of the underlying asset (annualized).
 * @param q      Continuous dividend yield.
 * @param n      Number of binomial steps.
 * @return The computed American option price.
 */
double binomial_crr_european_vanilla_option_cpu(const double S, const double K, const double T,
                                                const double r, const double sigma, const double q,
                                                const int n, const OptionType type);
