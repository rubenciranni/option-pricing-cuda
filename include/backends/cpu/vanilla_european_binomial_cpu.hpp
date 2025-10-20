#pragma once

#include "backends/cpu/vanilla_european_binomial_cpu.hpp"
#include "constants.hpp"

double vanilla_european_binomial_cpu(const double S, const double K, const double T, const double r,
                                     const double sigma, const double q, const int n,
                                     const OptionType type);
