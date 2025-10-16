
#include "functions_version.hpp"
#include "backends/cpu/binomial_crr_american_vanilla_option_cpu.hpp"
#include "backends/openmp/binomial_crr_american_vanilla_option_openmp.hpp"


std::map<std::string, PricingFunction> FUNCTIONS = {
    {"binomial_crr_american_vanilla_option_cpu", binomial_crr_american_vanilla_option_cpu},
    {"openmp", binomial_crr_american_vanilla_option_cpu_openmp},
    {"remove_zeros", binomial_crr_american_vanilla_option_cpu_remove_zeros}
};
