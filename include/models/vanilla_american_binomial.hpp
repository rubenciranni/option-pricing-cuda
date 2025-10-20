#pragma once

#include <string>

#include "constants.hpp"

/**
 * @brief Prices an American option using the Cox–Ross–Rubinstein (CRR) binomial model.
 *
 * @param S       Current underlying asset price.
 * @param K       Strike price of the option.
 * @param T       Time to maturity (in years).
 * @param r       Risk-free interest rate (annualized).
 * @param sigma   Volatility of the underlying asset (annualized).
 * @param q       Continuous dividend yield.
 * @param n       Number of binomial steps.
 * @param backend The backend to use for computation.
 * @return The computed American option price.
 */
float vanilla_american_binomial(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type, const Backend backend);
