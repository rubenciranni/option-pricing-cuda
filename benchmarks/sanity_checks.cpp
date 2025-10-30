#include "sanity_checks.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

std::vector<SingleRun> TESTS = {
    SingleRun(100, 100, 0.5, 0.03, 0.2, 0.015, 60, OptionType::Put, "At-the-money put"),
    SingleRun(150, 100, 1.0, 0.05, 0.25, 0.02, 100, OptionType::Put, "Deep OTM put"),
    SingleRun(50, 100, 1.0, 0.05, 0.25, 0.02, 100, OptionType::Put, "Deep ITM put"),
    SingleRun(100, 100, 0.01, 0.03, 0.2, 0.015, 30, OptionType::Put, "Short maturity put"),
    SingleRun(100, 100, 5.0, 0.05, 0.25, 0.02, 500, OptionType::Put, "Long maturity put"),
    SingleRun(100, 100, 1.0, 0.03, 0.8, 0.01, 200, OptionType::Put, "High volatility put"),
    SingleRun(100, 100, 1.0, 0.03, 0.01, 0.01, 200, OptionType::Put, "Low volatility put"),
    SingleRun(100, 100, 1.0, 0.0, 0.2, 0.01, 100, OptionType::Put, "Zero interest rate put"),
    SingleRun(100, 100, 1.0, -0.01, 0.25, 0.01, 100, OptionType::Put, "Negative interest rate put"),
    SingleRun(100, 100, 1.0, 0.03, 0.2, 0.0, 100, OptionType::Put, "Zero dividend put"),
    SingleRun(100, 100, 1.0, 0.03, 0.2, 0.1, 100, OptionType::Put, "High dividend yield put"),
    SingleRun(100, 100, 1.0, 0.03, 0.2, 0.01, 1, OptionType::Put, "Very small time step put"),
    SingleRun(1000, 100, 1.0, 0.03, 0.3, 0.01, 100, OptionType::Put, "Extreme spot vs strike put"),
    SingleRun(100, 100, 1.0, 0.0, 0.0, 0.0, 100, OptionType::Put, "Near zero vol and rate put"),
    SingleRun(100, 80, 0.05, 0.02, 0.6, 0.01, 25, OptionType::Put, "Short-term high vol put"),
    SingleRun(80, 100, 3.0, 0.05, 0.1, 0.02, 500, OptionType::Put, "Long-term low vol put"),
};
