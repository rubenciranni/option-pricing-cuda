#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

#include <algorithm>
#include <cmath>
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

int bin_search_zeros(const int n, const double S, const double K, const double u, const int sign) {
    int lower = 0;
    int upper = n;
    while (lower < upper - 1) {
        int mid = (upper + lower) / 2;
        double S_i_n = sign * (S * std::pow(u, mid * 2 - n) - K);
        if (S_i_n < 0) {
            upper = mid;
        } else {
            lower = mid;
        }
    }
    return lower;
}

int bin_search_red(const int n, const double S, const double K, const double u, const double up,
                   const double down, const int sign) {
    int lower = 0;
    int upper = n - 1;
    while (lower < upper - 1) {
        int mid = (upper + lower) / 2;
        double S_i_n = sign * (S * std::pow(u, mid * 2 - n) - K);
        double S_i1_n = sign * (S * std::pow(u, (mid + 1) * 2 - n) - K);
        double S_i_n_2 = sign * (S * std::pow(u, mid * 2 - (n - 1)) - K);
        if (S_i1_n < 0 || S_i_n_2 < 0 || S_i_n < 0) {
            upper = mid;
        } else if (up * S_i1_n + down * S_i_n < S_i_n_2) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    return lower;
}

double vanilla_american_binomial_cpu_trimotm(const double S, const double K, const double T,
                                             const double r, const double sigma, const double q,
                                             const int n, const OptionType type) {
    if (type == OptionType::Call) {
        return vanilla_american_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
    }

    const double deltaT = T / n;
    const double up = std::exp(sigma * std::sqrt(deltaT));

    const double disc = std::exp(-r * deltaT);
    const double p0 = (up * std::exp(-q * deltaT) - disc) / (up * up - 1.0);
    const double p1 = disc - p0;

    int sign = option_type_sign(type);
    int last_non_zero = bin_search_zeros(n, S, K, up, sign);

    std::vector<double> p(last_non_zero * 2 + 1);
    for (int i = 0; i <= std::min(last_non_zero, n); ++i) {
        double ST = S * std::pow(up, 2.0 * i - n);
        p[i] = sign * (ST - K);
    }
    for (int j = n - 1; j >= 0; --j) {
        for (int i = 0; i <= std::min(j, last_non_zero); ++i) {
            double ST = S * std::pow(up, 2.0 * i - j);

            double hold = p0 * p[i + 1] + p1 * p[i];
            double exercise = sign * (ST - K);
            p[i] = std::max(hold, exercise);
        }
    }

    return p[0];
}

double vanilla_american_binomial_cpu_trimotm_stprecomp(const double S, const double K,
                                                       const double T, const double r,
                                                       const double sigma, const double q,
                                                       const int n, const OptionType type) {
    if (type == OptionType::Call) {
        return vanilla_american_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
    }

    const double deltaT = T / n;
    const double u = std::exp(sigma * std::sqrt(deltaT));
    const double d = 1.0 / u;

    const double p = (exp((r - q) * deltaT) - d) / (u - d);
    const double risk_free_rate = std::exp(-r * deltaT);
    const double one_minus_p = 1.0 - p;
    const double up = p * risk_free_rate;
    const double down = one_minus_p * risk_free_rate;

    int sign = option_type_sign(type);

    int last_non_zero = bin_search_zeros(n, S, K, u, sign);
    std::vector<double> p_store(n * 2, 0);
    std::vector<double> s_store(n * 2, 0);
    for (int i = 0; i <= last_non_zero * 2; ++i) {
        s_store[i] = sign * (S * std::pow(u, i - n) - K);
    }
    auto get_index = [n](int i, int j) { return (2 * i - j) + n; };

    for (int i = 0; i <= last_non_zero; ++i) {
        p_store[i] = s_store[get_index(i, n)];
    }
    for (int j = n - 1; j >= 0; --j) {
        for (int i = 0; i <= j && i <= last_non_zero; ++i) {
            double S_i_j = s_store[get_index(i, j)];
            double hold = up * p_store[i + 1] + down * p_store[i];
            double exercise = S_i_j;
            p_store[i] = std::max(hold, exercise);
        }
    }
    return p_store[0];
}

double vanilla_american_binomial_cpu_trimotm_trimeeoff_stprecomp(const double S, const double K,
                                                                 const double T, const double r,
                                                                 const double sigma, const double q,
                                                                 const int n,
                                                                 const OptionType type) {
    if (type == OptionType::Call || n == 1) {
        return vanilla_american_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
    }
    const double deltaT = T / n;
    const double u = std::exp(sigma * std::sqrt(deltaT));
    const double d = 1.0 / u;

    const double p = (exp((r - q) * deltaT) - d) / (u - d);
    const double risk_free_rate = std::exp(-r * deltaT);
    const double one_minus_p = 1.0 - p;
    const double up = p * risk_free_rate;
    const double down = one_minus_p * risk_free_rate;

    int sign = option_type_sign(type);

    int last_non_zero = bin_search_zeros(n, S, K, u, sign);
    int red_last_index = std::max(bin_search_red(n, S, K, u, up, down, sign), 0);

    std::vector<double> p_store(n * 2, 0);
    std::vector<double> s_store(n * 2, 0);
    for (int i = 0; i <= last_non_zero * 2; ++i) {
        s_store[i] = sign * (S * std::pow(u, i - n) - K);
    }
    auto get_index = [n](int i, int j) { return (2 * i - j) + n; };

    for (int i = 0; i <= last_non_zero; ++i) {
        p_store[i] = s_store[get_index(i, n)];
    }
    for (int j = n - 1; j >= 0; --j) {
        for (int i = std::max(red_last_index, 0); i <= j && i <= last_non_zero; ++i) {
            double S_i_j = s_store[get_index(i, j)];
            double hold = up * p_store[i + 1] + down * p_store[i];
            double exercise = S_i_j;
            p_store[i] = std::max(hold, exercise);
        }
        if (red_last_index >= 0) {
            p_store[red_last_index] = s_store[get_index(red_last_index, j)];
            red_last_index--;
        }
    }
    return p_store[0];
}

double vanilla_american_binomial_cpu_trimotm_trimeeon_stprecomp(const double S, const double K,
                                                                const double T, const double r,
                                                                const double sigma, const double q,
                                                                const int n,
                                                                const OptionType type) {
    if (type == OptionType::Call || n == 1) {
        return vanilla_american_binomial_cpu_naive(S, K, T, r, sigma, q, n, type);
    }
    const double deltaT = T / n;
    const double u = std::exp(sigma * std::sqrt(deltaT));
    const double d = 1.0 / u;

    const double p = (exp((r - q) * deltaT) - d) / (u - d);
    const double risk_free_rate = std::exp(-r * deltaT);
    const double one_minus_p = 1.0 - p;
    const double up = p * risk_free_rate;
    const double down = one_minus_p * risk_free_rate;

    int sign = option_type_sign(type);

    int last_non_zero = bin_search_zeros(n, S, K, u, sign);
    int red_last_index = std::max(bin_search_red(n, S, K, u, up, down, sign), 0);

    std::vector<double> p_store(n * 2, 0);
    std::vector<double> s_store(n * 2, 0);
    for (int i = 0; i <= last_non_zero * 2; ++i) {
        s_store[i] = sign * (S * std::pow(u, i - n) - K);
    }
    auto get_index = [n](int i, int j) { return (2 * i - j) + n; };

    for (int i = 0; i <= last_non_zero; ++i) {
        p_store[i] = s_store[get_index(i, n)];
    }
    for (int j = n - 1; j >= 0; --j) {
        for (int i = std::max(red_last_index, 0); i <= j && i <= last_non_zero; ++i) {
            double S_i_j = s_store[get_index(i, j)];
            double hold = up * p_store[i + 1] + down * p_store[i];
            double exercise = S_i_j;
            p_store[i] = std::max(hold, exercise);
            if (hold < exercise && i < red_last_index) {
                red_last_index = i;
            }
        }
        if (red_last_index >= 0) {
            p_store[red_last_index] = s_store[get_index(red_last_index, j)];
            red_last_index--;
        }
    }

    // return std::max(p_store[0],s_store[get_index(0,0)]);
    return p_store[0];
}
