#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

double vanilla_american_binomial_cpu_naive(const double S, const double K, const double T,
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
    double ST = S * std::pow(up, 2.0 * i - n);
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

double vanilla_american_binomial_cpu_remove_zeros(const double S, const double K, const double T,
                                                  const double r, const double sigma,
                                                  const double q, const int n,
                                                  const OptionType type) {
  const double deltaT = T / n;
  const double up = std::exp(sigma * std::sqrt(deltaT));

  const double disc = std::exp(-r * deltaT);
  const double p0 = (up * std::exp(-q * deltaT) - disc) / (up * up - 1.0);
  const double p1 = disc - p0;

  int lower = 0;
  int upper = n * 2;
  while (lower < upper - 1) {
    int mid = (upper + lower) / 2;
    double S_i_n = -1 * (S * std::pow(up, mid - n) - K);
    if (S_i_n < 0) {
      upper = mid;
    } else {
      lower = mid;
    }
  }
  int last_non_zero = lower;

  int sign = option_type_sign(type);
  if (sign > 0) {
    // TODO: implement the other direction (call)
    return 0.0;
  }
  std::vector<double> p(last_non_zero + 1);
  for (int i = 0; i * 2 <= std::min(last_non_zero, n * 2); ++i) {
    double ST = S * std::pow(up, 2.0 * i - n);
    p[i] = sign * (ST - K);
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int i = 0; i * 2 <= std::min(j * 2, last_non_zero); ++i) {
      double ST = S * std::pow(up, 2.0 * i - j);

      double hold = p0 * p[i + 1] + p1 * p[i];
      double exercise = sign * (ST - K);
      p[i] = std::max(hold, exercise);
    }
  }

  return p[0];
}

double vanilla_american_binomial_cpu_remove_zeros_cache(const double S, const double K,
                                                        const double T, const double r,
                                                        const double sigma, const double q,
                                                        const int n, const OptionType type) {
  const double deltaT = T / n;
  const double u = std::exp(sigma * std::sqrt(deltaT));
  const double d = 1.0 / u;

  const double p = (exp((r - q) * deltaT) - d) / (u - d);
  const double risk_free_rate = std::exp(-r * deltaT);
  const double one_minus_p = 1.0 - p;
  const double up = p * risk_free_rate;
  const double down = one_minus_p * risk_free_rate;

  int sign = option_type_sign(type);

  if (sign != -1) {
    // TODO: implement the other direction (call)
    return 0.0;
  }
  int lower = 0;
  int upper = n * 2;
  while (lower < upper - 1) {
    int mid = (upper + lower) / 2;
    double S_i_n = -1 * (S * std::pow(u, mid - n) - K);
    if (S_i_n < 0) {
      upper = mid;
    } else {
      lower = mid;
    }
  }
  int last_non_zero = lower;
  std::vector<double> p_store((last_non_zero + 3) / 2, 0);
  std::vector<double> s_store(n * 2, 0);
  for (int i = 0; i <= last_non_zero; ++i) {
    s_store[i] = -1 * (S * std::pow(u, i - n) - K);
  }
  auto get_index = [n](int i, int j) { return (2 * i - j) + n; };

  for (int i = 0; i * 2 <= last_non_zero; ++i) {
    p_store[i] = s_store[get_index(i, n)];
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int i = 0; i <= j && (i * 2) <= last_non_zero; ++i) {
      double S_i_j = s_store[get_index(i, j)];
      double hold = up * p_store[i + 1] + down * p_store[i];
      double exercise = S_i_j;
      p_store[i] = std::max(hold, exercise);
    }
  }
  return p_store[0];
}
