#include "models/vanilla_european_binomial.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "backends/cpu/vanilla_european_binomial_cpu.hpp"

float vanilla_european_binomial(const double S, const double K, const double T, const double r,
                                const double sigma, const double q, const int n,
                                const OptionType type, const Backend backend) {
  if (backend == Backend::CPU) {
    return vanilla_european_binomial_cpu(S, K, T, r, sigma, q, n, type);
  } else {
    throw std::invalid_argument("Unknown backend: " + to_string(backend));
  }
}
