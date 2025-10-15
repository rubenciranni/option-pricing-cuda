#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include "constants.hpp"
#include <omp.h>

double binomial_crr_american_vanilla_option_cpu(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type) {
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
