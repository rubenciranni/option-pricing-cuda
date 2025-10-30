#include "functions_version.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/openmp/vanilla_american_binomial_openmp.hpp"

std::map<std::string, PricingFunction> FUNCTIONS = {
    {"vanilla_american_binomial_cpu_naive", vanilla_american_binomial_cpu_naive},
    {"vanilla_american_binomial_cpu_remove_zeros_cache",
     vanilla_american_binomial_cpu_remove_zeros_cache},
    {"vanilla_american_binomial_openmp_naive", vanilla_american_binomial_openmp_naive},
    {"vanilla_american_binomial_cpu_remove_zeros", vanilla_american_binomial_cpu_remove_zeros},
    {"vanilla_american_binomial_cuda_naive", vanilla_american_binomial_cuda_naive},
    {"vanilla_american_binomial_cuda_no_sync", vanilla_american_binomial_cuda_no_sync},
    {"vanilla_american_binomial_cuda_fill", vanilla_american_binomial_cuda_fill},
    {"vanilla_american_binomial_cuda_tile", vanilla_american_binomial_cuda_tile},
    {"vanilla_american_binomial_cuda_unroll", vanilla_american_binomial_cuda_unroll},
    {"vanilla_american_binomial_cuda_precomputed_stock_price",
     vanilla_american_binomial_cuda_precomputed_stock_price},
    {"vanilla_american_binomial_cuda_precomputed_payoffs",
     vanilla_american_binomial_cuda_precomputed_payoffs}};
