#include <cmath>
#include <vector>

#include "catch.hpp"
#include "constants.hpp"
#include "models/vanilla_american_binomial.hpp"

TEST_CASE("American Option Put-Call Parity Approximation", "[ame_put_call_parity]") {
    double S = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double q = 0.0;  // no dividends
    int n = 500;

    for (auto backend : {Backend::CPU, Backend::OpenMP, Backend::CUDA}) {
        double putPrice =
            vanilla_american_binomial(S, K, T, r, sigma, q, n, OptionType::Put, backend);
        double callPrice =
            vanilla_american_binomial(S, K, T, r, sigma, q, n, OptionType::Call, backend);

        double callPriceParity = putPrice + S - K * std::exp(-r * T);

        INFO("Put Price: " << putPrice);
        INFO("Call Price: " << callPrice);
        INFO("Call Price from Parity: " << callPriceParity);

        REQUIRE(std::abs(callPrice - callPriceParity) < 1.0);
    }
}

TEST_CASE("American Option Linear Homogeneity", "[ame_linear_hom]") {
    double S = 100.0;
    double K = 150.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double q = 0.01;
    int n = 500;
    double alpha = 1.92767;

    for (auto backend : {Backend::CPU, Backend::OpenMP, Backend::CUDA}) {
        double callPrice =
            vanilla_american_binomial(S, K, T, r, sigma, q, n, OptionType::Call, backend);
        double callPriceS = vanilla_american_binomial(alpha * S, alpha * K, T, r, sigma, q, n,
                                                      OptionType::Call, backend);

        REQUIRE(std::abs(alpha * callPrice - callPriceS) < 1.0e-6);
    }
}
