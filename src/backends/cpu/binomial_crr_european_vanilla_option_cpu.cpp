#include "backends/cpu/binomial_crr_european_vanilla_option_cpu.hpp"
#include <math.h>
#include <vector>
#include <iostream>

double binomial_crr_european_vanilla_option_cpu(
    const double S, 
    const double K, 
    const double T,
    const double r, 
    const double sigma, 
    const double q, 
    const int n, 
    const OptionType type) {

    double dt = T / static_cast<double>(n);
    double disc = std::exp(-r * T);
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp((r-q) * dt) - d) / (u - d);

    int sign = option_type_sign(type);

    std::vector<double> log_fact(n+1);
    log_fact[0] = 0.0;
    for (int i=1;i<=n;i++) {
        log_fact[i] = log_fact[i-1]+std::log(i);
    }

    std::vector<double> log_binom(n+1);
    double logp = std::log(p);
    double logq = std::log(1.0-p);
    for (int i=0;i<=n;i++) {
        log_binom[i] = i*logp + (n-i)*logq + log_fact[n] - log_fact[i] - log_fact[n-i];
    }

    double price = 0.0;
    for (int i=0;i<=n;i++) {
        double ST = sign * (S * std::pow(u, 2*i - n) - K);
        double b = disc*std::exp(log_binom[i]);
        price += b*std::max(ST, 0.0);
    }
    return price;
}
