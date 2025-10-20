#include "backends/openmp/vanilla_american_binomial_openmp.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

double vanilla_american_binomial_openmp(const double S, const double K, const double T,
                                        const double r, const double sigma, const double q,
                                        const int n, const OptionType type) {
  const double deltaT = T / n;
  const double up = std::exp(sigma * std::sqrt(deltaT));

  const double disc = std::exp(-r * deltaT);
  const double p0 = (up * std::exp(-q * deltaT) - disc) / (up * up - 1.0);
  const double p1 = disc - p0;
  double* buf1 = new double[n + 1];
  double* buf2 = new double[n + 1];
  double* current = buf1;
  double* next = buf2;
  int sign = option_type_sign(type);
#pragma omp parallel for
  for (int i = 0; i <= n; ++i) {
    double ST = S * std::pow(up, 2.0 * i - n + 1.0);
    current[i] = std::max(sign * (ST - K), 0.0);
  }
  for (int j = n - 1; j >= 0; --j) {
#pragma omp parallel for
    for (int i = 0; i <= j; ++i) {
      double ST = S * std::pow(up, 2.0 * i - j);
      double hold = p0 * current[i + 1] + p1 * current[i];
      double exercise = std::max(sign * (ST - K), 0.0);
      next[i] = std::max(hold, exercise);
    }
    std::swap(current, next);
  }
  double result = current[0];
  delete[] buf1;
  delete[] buf2;
  return result;
}
