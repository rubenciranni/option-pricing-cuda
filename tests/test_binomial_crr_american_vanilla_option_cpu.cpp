#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "constants.hpp"
#include "binomial_crr_american_vanilla_option_cpu.hpp"
#include <cmath>

TEST_CASE("American Option Put-Call Parity Approximation", "[put_call_parity]") {
    double S = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double q = 0.0; // no dividends
    int n = 500;

    double putPrice = binomial_crr_american_vanilla_option_cpu(T, S, K, r, sigma, q, n, OptionType::Put);
    double callPrice = binomial_crr_american_vanilla_option_cpu(T, S, K, r, sigma, q, n, OptionType::Call);

    double callPriceParity = putPrice + S - K * std::exp(-r * T);

    INFO("Put Price: " << putPrice);
    INFO("Call Price: " << callPrice);
    INFO("Call Price from Parity: " << callPriceParity);

    REQUIRE(std::abs(callPrice - callPriceParity) < 1.0);
}
