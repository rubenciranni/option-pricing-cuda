#pragma once

#include "constants.hpp"

double vanilla_american_binomial_openmp(const double S, const double K, const double T,
                                        const double r, const double sigma, const double q,
                                        const int n, const OptionType type);
