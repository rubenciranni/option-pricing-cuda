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
    // TODO: implement the other direction
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

inline int ilog2(unsigned int x) {
  return 31 - __builtin_clz(x);
}
double vanilla_american_binomial_cpu_removed_zeros_and_red(const double S, const double K,
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
    // TODO: implement the other direction
    return 0.0;
  }

  int max_power = ilog2(n) + 1;
  double h_u_power_cache[max_power];
  h_u_power_cache[0] = 1.0;
  h_u_power_cache[1] = u;
  for (int i = 2; i <= max_power; ++i) {
    h_u_power_cache[i] = h_u_power_cache[i - 1] * h_u_power_cache[i - 1];
  }
  auto get_u_power = [&h_u_power_cache](int i) {
    double result = 1.0;
    int bit_pos = 1;
    bool is_negative = i < 0;
    if (is_negative) {
      i = -i;
    }
    while (i) {
      // is equal to int pos  = (i & 1) ? bit_pos : 0;
      //  cause h_u_power_cache[0] = 1.0
      int pos = (i & 1) * bit_pos;
      result *= h_u_power_cache[pos];
      i >>= 1;
      bit_pos++;
    }
    if (is_negative) {
      result = 1.0 / result;
    }
    return result;
  };

  int lower = 0;
  int upper = n * 2;
  while (lower < upper - 1) {
    int mid = (upper + lower) / 2;
    double S_i_n = -1 * (S * get_u_power(mid - n) - K);
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

  // last red
  upper = last_non_zero;
  lower = 0;
  while (lower < upper - 1) {
    int mid = (upper + lower) / 2;
    double S_i_n = s_store[get_index(mid, n - 1)];
    double hold = up * s_store[get_index(mid + 1, n)] + down * s_store[get_index(mid, n)];
    if (hold < S_i_n) {
      lower = mid;
    } else {
      upper = mid;
    }
  }
  int last_red = lower - 1;
  // this part is broken
  // for (int i = std::max(last_red-1,0)*2; i*2 <= last_non_zero; ++i) {
  for (int i = 0; i * 2 <= last_non_zero; ++i) {
    p_store[i] = s_store[get_index(i, n)];
  }

  // this part is broken
  // for (int i = std::max(last_red-1,0)*2; i*2 <= last_non_zero; ++i) {
  // int start = std::max(last_red, 0);
  // for (int j = n - 1; j >= 0; --j) {
  //   if(last_red >= 0){
  //     p_store[last_red] = s_store[get_index(last_red, j+1)];
  //   }
  //   for (int i = start; i <= j && (i*2) <=last_non_zero; ++i) {
  //     double exercise =  s_store[get_index(i, j)];
  //     double p_up =  p_store[i + 1];
  //     double p_down =  p_store[i] ;
  //     double hold = up *  p_store[i + 1]  + down * p_store[i];
  //     if (hold < exercise) {
  //       p_store[i] =  exercise;
  //     }else{
  //       p_store[i] = hold;
  //     }
  //   }
  //   start = std::max(--last_red, 0);
  // }
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
