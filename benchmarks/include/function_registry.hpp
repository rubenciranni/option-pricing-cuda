#pragma once

#include <functional>
#include <map>
#include <string>

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/openmp/vanilla_american_binomial_openmp.hpp"
#include "constants.hpp"


typedef double (*PricingFunction)(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type);

extern std::map<std::string, PricingFunction> FUNCTION_REGISTRY;
