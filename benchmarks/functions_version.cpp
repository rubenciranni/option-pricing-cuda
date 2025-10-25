
#include "functions_version.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/openmp/vanilla_american_binomial_openmp.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"

std::map<std::string, PricingFunction> FUNCTIONS = {
    // {"vanilla_american_binomial_cpu", vanilla_american_binomial_cpu},
    // {"vanilla_american_binomial_cpu_remove_zeros_cache",
    //  vanilla_american_binomial_cpu_remove_zeros_cache},
    // {"vanilla_american_binomial_openmp", vanilla_american_binomial_openmp},
    {"vanilla_american_binomial_cpu_index_opt_cache",
     vanilla_american_binomial_cpu_index_opt_cache},
    // {"vanilla_american_binomial_cpu_remove_zeros", vanilla_american_binomial_cpu_remove_zeros},
    {"vanilla_american_binomial_cuda", vanilla_american_binomial_cuda},
    {"vanilla_american_binomial_cuda_no_sync",vanilla_american_binomial_cuda_no_sync}
};
