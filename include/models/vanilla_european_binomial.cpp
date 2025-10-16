#include <string>

#include "constants.hpp"
float vanilla_european_binomial(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type, const std::string backend);
