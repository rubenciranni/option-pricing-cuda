#include "constants.hpp"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include "constants.hpp"



double binomial_crr_american_vanilla_option_cpu_openmp(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type) ;