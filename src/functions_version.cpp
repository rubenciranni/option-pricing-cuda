
#include "functions_version.hpp"


std::map<std::string, PricingFunction> FUNCTIONS = {
    {"binomial_crr_american_vanilla_option_cpu", binomial_crr_american_vanilla_option_cpu},
    {"openmp", binomial_crr_american_vanilla_option_cpu_openmp},
};
