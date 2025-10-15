#pragma once
#include "constants.hpp"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>



double binomial_crr_american_vanilla_option_cpu_remove_zeros(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type) ;