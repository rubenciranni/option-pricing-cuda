#include "backends/cpu/binomial_crr_american_vanilla_option_cpu.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

double binomial_crr_american_vanilla_option_cpu(const double S, const double K, const double T,
                                                const double r, const double sigma, const double q,
                                                const int n, const OptionType type) {
  const double deltaT = T / n;
  const double up = std::exp(sigma * std::sqrt(deltaT));

  const double disc = std::exp(-r * deltaT);
  const double p0 = (up * std::exp(-q * deltaT) - disc) / (up * up - 1.0);
  const double p1 = disc - p0;

  std::vector<double> p(n + 1);
  int sign = option_type_sign(type);
  for (int i = 0; i <= n; ++i) {
    double ST = S * std::pow(up, 2.0 * i - n + 1.0);
    p[i] = std::max(sign * (ST - K), 0.0);
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int i = 0; i <= j; ++i) {
      double ST = S * std::pow(up, 2.0 * i - j);
      double hold = p0 * p[i + 1] + p1 * p[i];
      double exercise = std::max(sign * (ST - K), 0.0);
      p[i] = std::max(hold, exercise);
    }
  }

  return p[0];
}

double binomial_crr_american_vanilla_option_cpu_remove_zeros(const double S, const double K,
                                                             const double T, const double r,
                                                             const double sigma, const double q,
                                                             const int n, const OptionType type) {
  const double deltaT = T / n;
  const double up = std::exp(sigma * std::sqrt(deltaT));

  const double disc = std::exp(-r * deltaT);
  const double p0 = (up * std::exp(-q * deltaT) - disc) / (up * up - 1.0);
  const double p1 = disc - p0;
  int round_up =
      std::round((n - 1) / 2 +
                 0.5 * std::log(K / S) /
                     std::log(up));  // you can find it from the formula   0 < K - S * up^(2i - n)

  std::vector<double> p(n + 1);
  int sign = option_type_sign(type);
  // call
  if (sign < 0) {
    for (int i = 0; i <= round_up; ++i) {
      double ST = S * std::pow(up, 2.0 * i - n + 1.0);
      p[i] = std::max(sign * (ST - K), 0.0);
      // are the non zero
    }
  } else {
    for (int i = round_up + 1; i <= n; ++i) {
      double ST = S * std::pow(up, 2.0 * i - n + 1.0);
      p[i] = std::max(sign * (ST - K), 0.0);
    }
  }
  if (sign < 0) {
    for (int j = n - 1; j >= 0; --j) {
      for (int i = 0; i <= std::min(j, round_up); ++i) {
        double ST = S * std::pow(up, 2.0 * i - j);
        double hold = p0 * p[i + 1] + p1 * p[i];
        double exercise = std::max(sign * (ST - K), 0.0);
        p[i] = std::max(hold, exercise);
      }
    }
  } else {
    for (int j = n - 1; j >= 0; --j) {
      for (int i = std::max(0, round_up - (n - 1 - j)); i <= j; ++i) {
        double ST = S * std::pow(up, 2.0 * i - j);
        double hold = p0 * p[i + 1] + p1 * p[i];
        double exercise = std::max(sign * (ST - K), 0.0);
        p[i] = std::max(hold, exercise);
      }
    }
  }

  return p[0];
}
