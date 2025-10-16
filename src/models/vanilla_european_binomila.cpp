#include <math.h>

#include <iostream>
#include <vector>

#include "backends/cpu/binomial_crr_european_vanilla_option_cpu.hpp"
#include "models/vanilla_european_binomial.hpp"

float vanilla_european_binomial(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type, const std::string backend) {
  if (backend == "cpu") {
    return binomial_crr_european_vanilla_option_cpu(S, K, T, r, sigma, q, n, type);
  } else {
    throw std::invalid_argument("Unknown backend: " + backend);
  }
}
