#include "models/vanilla_american_binomial.hpp"

#include <iostream>

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/openmp/vanilla_american_binomial_openmp.hpp"
#include "constants.hpp"

float vanilla_american_binomial(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type, const Backend backend) {
    if (backend == Backend::CPU) {
        return vanilla_american_binomial_cpu(S, K, T, r, sigma, q, n, type);
    } else if (backend == Backend::OpenMP) {
        return vanilla_american_binomial_openmp(S, K, T, r, sigma, q, n, type);
    } else if (backend == Backend::CUDA) {
        return vanilla_american_binomial_cuda(S, K, T, r, sigma, q, n, type);
    } else {
        throw std::invalid_argument("Unknown backend: " + to_string(backend));
    }
}
