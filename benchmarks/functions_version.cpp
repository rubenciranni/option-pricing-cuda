
#include "functions_version.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/openmp/vanilla_american_binomial_openmp.hpp"

std::map<std::string, PricingFunction> FUNCTIONS = {
    {"vanilla_american_binomial_cpu", vanilla_american_binomial_cpu},
    {"vanilla_american_binomial_cpu_remove_zeros_cache",
     vanilla_american_binomial_cpu_remove_zeros_cache},
    {"vanilla_american_binomial_openmp", vanilla_american_binomial_openmp},
    {"vanilla_american_binomial_cpu_remove_zeros", vanilla_american_binomial_cpu_remove_zeros}};
