#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
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
double vanilla_american_binomial_cpu_remove_zeros_cache(const double S, const double K,
                                                        const double T, const double r,
                                                        const double sigma, const double q,
                                                        const int n, const OptionType type) {
  const double deltaT = T / n;
  const double u = std::exp(sigma * std::sqrt(deltaT));
  const double d = 1.0 / u;

  // const double up_old = std::exp(sigma * std::sqrt(deltaT));
  // const double disc_old = std::exp(-r * deltaT);
  // const double p0_old = (up_old * std::exp(-q * deltaT) - disc_old) / (up_old * up_old - 1.0);
  // const double p1_old = disc_old - p0_old;

  const double p = (exp((r - q) * deltaT) - d) / (u - d);
  const double risk_free_rate = std::exp(-r * deltaT);
  const double one_minus_p = 1.0 - p;
  const double up = p * risk_free_rate;
  const double down = one_minus_p * risk_free_rate;
  // std::cout << "up: " << up << " down: " << down << std::endl;
  // std::cout << "p0_old: " << p0_old << " p1_old: " << p1_old << std::endl;

  int round_up = std::round((n - 1) / 2 + 0.5 * std::log(K / S) / std::log(u)) +
                 1;  // you can find it from the formula   0 < K - S * up^(2i - n)

  int sign = option_type_sign(type);
  std::vector<double> p_store(n + 1);
  std::vector<double> s_store(2 * n + 1);
  // std::cout <<"n: " << n << std::endl;

  if (sign != -1) {
    // std::cout << ("Not implemented for options\n");
    return 0.0;
  }
  // std::cout <<"round_up: " << round_up << std::endl;
  for (int i = 0; i <= n * 2; ++i) {
    s_store[i] = -1 * (S * std::pow(u, i - n) - K);
    // std::cout << "i: " << i <<" power "<< i - n + 1 << " value: " << s_store[i] << std::endl;
  }

  auto get_index = [n](int i, int j) { return (2 * i - j) + n; };
  for (int i = 0; i <= round_up; ++i) {
    p_store[i] = s_store[get_index(i, n)];
    // std::cout << "p_store[" << i << "] = " << p_store[i]<< " index" <<get_index(i,n) - (n-1)<<
    // std::endl;
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int i = 0; i <= std::min(j, round_up); ++i) {
      // std::cout << "j: " << j << " i: " << i << " index: " << get_index(i, j) - (n-1) <<
      // std::endl;
      double S_i_j = s_store[get_index(i, j)];
      double hold = up * p_store[i + 1] + down * p_store[i];
      double exercise = S_i_j;
      p_store[i] = std::max(hold, exercise);
      // if (j > n - 3)
      //   std::cout << "j: " << j << " i: " << i << " red "<< p_store[i]<< std::endl;
    }
    // std::cout << std::endl;
  }
  // std::cout << "Final price: " << p_store[0] << std::endl;
  return p_store[0];
}
