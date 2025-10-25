#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

double vanilla_american_binomial_cpu(const double S, const double K, const double T, const double r,
                                     const double sigma, const double q, const int n,
                                     const OptionType type) {
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

double vanilla_american_binomial_cpu_remove_zeros(const double S, const double K, const double T,
                                                  const double r, const double sigma,
                                                  const double q, const int n,
                                                  const OptionType type) {
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

  int round_up = std::round((n - 1) / 2 + 0.5 * std::log(K / S) / std::log(u)) +
                 1;  // you can find it from the formula   0 < K - S * up^(2i - n)

  int sign = option_type_sign(type);
  std::vector<double> p_store(n + 1);
  std::vector<double> s_store(2 * n + 1);

  if (sign != -1) {
    // TODO: implement the other direction
    return 0.0;
  }
  for (int i = 0; i <= n * 2; ++i) {
    s_store[i] = -1 * (S * std::pow(u, i - n) - K);
  }

  auto get_index = [n](int i, int j) { return (2 * i - j) + n; };
  for (int i = 0; i <= round_up; ++i) {
    p_store[i] = s_store[get_index(i, n)];
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int i = 0; i <= std::min(j, round_up); ++i) {
      double S_i_j = s_store[get_index(i, j)];
      double hold = up * p_store[i + 1] + down * p_store[i];
      double exercise = S_i_j;
      p_store[i] = std::max(hold, exercise);
    }
  }
  return p_store[0];
}

double vanilla_american_binomial_cpu_index_opt_cache_old(const double S, const double K,
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

  // you can find it from the formula   0 < K - S * up^(2i - n)
  int round_up = std::round((n - 1) / 2 + 0.5 * std::log(K / S) / std::log(u)) + 1;

  int sign = option_type_sign(type);
  std::vector<double> p_store(n + 1);
  std::vector<double> s_store(2 * n + 1);

  if (sign != -1) {
    // TODO: implement the other direction
    return 0.0;
  }
  for (int i = 0; i <= n * 2; ++i) {
    s_store[i] = -1 * (S * std::pow(u, i - n) - K);
  }

  auto get_index = [n](int i, int j) { return (2 * i - j) + n; };
  for (int i = 0; i <= round_up; ++i) {
    p_store[i] = s_store[get_index(i, n)];
  }

  int last_red = -1;

  for (int j = n - 1; j >= 0; --j) {
    bool blue_found = false;
    for (int i = std::max(last_red, 0); i <= std::min(j, round_up); ++i) {
      double S_i_j = s_store[get_index(i, j)];
      double up_value = (round_up + 1) == i ? s_store[get_index(i + 1, j + 1)] : p_store[i + 1];
      double down_value = i == last_red ? s_store[get_index(i, j + 1)] : p_store[i];

      double hold = up * up_value + down * down_value;
      double exercise = S_i_j;
      if (hold < exercise) {
        p_store[i] = exercise;
        if (!blue_found) last_red = i - 1;
      } else {
        p_store[i] = hold;
        if (!blue_found) {
          blue_found = true;
        }
      }
    }
  }
  return p_store[0];
}
double vanilla_american_binomial_cpu_index_opt_cache(const double S, const double K, const double T,
                                                     const double r, const double sigma,
                                                     const double q, const int n,
                                                     const OptionType type) {
  const double deltaT = T / n;
  const double u = std::exp(sigma * std::sqrt(deltaT));
  const double d = 1.0 / u;
  const double p = (exp((r - q) * deltaT) - d) / (u - d);
  const double risk_free_rate = std::exp(-r * deltaT);
  const double one_minus_p = 1.0 - p;
  const double up = p * risk_free_rate;
  const double down = one_minus_p * risk_free_rate;

  // you can find it from the formula   0 < K - S * up^(2i - n)
  int round_up = std::round((n - 1) / 2 + 0.5 * std::log(K / S) / std::log(u)) + 1;

  int sign = option_type_sign(type);
  std::vector<double> p_store(n + 1);
  std::vector<double> s_store(2 * n + 1);

  if (sign != -1) {
    // TODO: implement the other direction
    return 0.0;
  }
  for (int i = 0; i <= n * 2; ++i) {
    s_store[i] = -1 * (S * std::pow(u, i - n) - K);
  }

  auto get_index = [n](int i, int j) { return (2 * i - j) + n; };
  for (int i = 0; i <= round_up; ++i) {
    p_store[i] = s_store[get_index(i, n)];
  }

  int last_red = -1;

  for (int j = n - 1; j >= 0; --j) {
    bool blue_found = false;
    int i = std::max(last_red, 0);
    double S_i_j = s_store[get_index(i, j)];
    double up_value = (round_up + 1) == i ? s_store[get_index(i + 1, j + 1)] : p_store[i + 1];
    double down_value = i == last_red ? s_store[get_index(i, j + 1)] : p_store[i];
    double hold = up * up_value + down * down_value;
    if (hold < S_i_j) {
      p_store[i] = S_i_j;
      last_red = i - 1;
    } else {
      p_store[i] = hold;
      blue_found = true;
    }
    i++;
    for (; i <= std::min(j, round_up) - 1 && !blue_found; ++i) {
      S_i_j = s_store[get_index(i, j)];
      hold = up * p_store[i + 1] + down * p_store[i];
      if (hold < S_i_j) {
        p_store[i] = S_i_j;
        last_red = i - 1;
      } else {
        p_store[i] = hold;
        blue_found = true;
      }
    }
    for (; i <= std::min(j, round_up) - 1 && blue_found; ++i) {
      S_i_j = s_store[get_index(i, j)];
      hold = up * p_store[i + 1] + down * p_store[i];
      p_store[i] = std::max(hold, S_i_j);
    }
    for (; i <= std::min(j, round_up); ++i) {
      double S_i_j = s_store[get_index(i, j)];
      double up_value = (round_up + 1) == i ? s_store[get_index(i + 1, j + 1)] : p_store[i + 1];
      double down_value = i == last_red ? s_store[get_index(i, j + 1)] : p_store[i];
      hold = up * up_value + down * down_value;
      p_store[i] = std::max(hold, S_i_j);
    }
  }
  return p_store[0];
}
