#include "backends/openmp/binomial_crr_american_vanilla_option_openmp.hpp"
#include "backends/cpu/binomial_crr_american_vanilla_option_cpu.hpp"
#include "models/vanilla_american_binomial.hpp"
#include <iostream>



float vanilla_american_binomial(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type,const std::string backend){
    if (backend == "cpu"){
        return binomial_crr_american_vanilla_option_cpu_remove_zeros(S, K, T, r, sigma, q, n, type);
    }
    else if (backend == "openmp"){
        return binomial_crr_american_vanilla_option_cpu_openmp(S, K, T, r, sigma, q, n, type);
    }
    else{
        throw std::invalid_argument("Unknown backend: " + backend);
    }

}