#pragma once

#include <vector>

#include "benchmark_parameters.hpp"
#include "function_registry.hpp"

// clang-format off
inline std::vector<PricingInput> SANITY_CHECK_PRICING_INPUTS = {
    PricingInput(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, OptionType::Put, "At-the-money put"),
    PricingInput(150, 100, 1.0, 0.05, 0.25, 0.02, 100, OptionType::Put, "Deep OTM put"),
    PricingInput(50, 100, 1.0, 0.05, 0.25, 0.02, 100, OptionType::Put, "Deep ITM put"),
    PricingInput(100, 100, 0.01, 0.03, 0.2, 0.015, 30, OptionType::Put, "Short maturity put"),
    PricingInput(100, 100, 5.0, 0.05, 0.25, 0.02, 500, OptionType::Put, "Long maturity put"),
    PricingInput(100, 100, 1.0, 0.03, 0.8, 0.01, 200, OptionType::Put, "High volatility put"),
    PricingInput(100, 100, 1.0, 0.03, 0.01, 0.01, 200, OptionType::Put, "Low volatility put"),
    PricingInput(100, 100, 1.0, 0.0, 0.2, 0.01, 100, OptionType::Put, "Zero interest rate put"),
    PricingInput(100, 100, 1.0, -0.01, 0.25, 0.01, 100, OptionType::Put, "Negative interest rate put"),
    PricingInput(100, 100, 1.0, 0.03, 0.2, 0.0, 100, OptionType::Put, "Zero dividend put"),
    PricingInput(100, 100, 1.0, 0.03, 0.2, 0.1, 100, OptionType::Put, "High dividend yield put"),
    PricingInput(100, 100, 1.0, 0.03, 0.2, 0.01, 1, OptionType::Put, "Very small time step put"),
    PricingInput(1000, 100, 1.0, 0.03, 0.3, 0.01, 100, OptionType::Put, "Extreme spot vs strike put"),
    PricingInput(100, 100, 1.0, 0.0, 0.0, 0.0, 100, OptionType::Put, "Near zero vol and rate put"),
    PricingInput(100, 80, 0.05, 0.02, 0.6, 0.01, 25, OptionType::Put, "Short-term high vol put"),
    PricingInput(80, 100, 3.0, 0.05, 0.1, 0.02, 500, OptionType::Put, "Long-term low vol put"),
};
// clang-format on

typedef std::vector<std::tuple<PricingInput, double, double>> SanityCheckResults;

class SanityChecker {
   public:
    std::string reference_function_name;
    PricingFunction reference_function;
    std::vector<std::pair<PricingInput, double>> reference_function_results;
    bool with_logging = true;
    bool builded = false;

    SanityChecker(std::string name, PricingFunction function)
        : reference_function_name(name), reference_function(function), builded(false) {
        reference_function_results = std::vector<std::pair<PricingInput, double>>();
        for (const auto& run : SANITY_CHECK_PRICING_INPUTS) {
            double price =
                reference_function(run.S, run.K, run.T, run.r, run.sigma, run.q, run.n, run.type);
            reference_function_results.push_back(std::make_pair(run, price));
        }
    }

    SanityCheckResults run_single_all_sanity_checks(PricingFunction func) {
        SanityCheckResults test_results;
        for (const auto& test : reference_function_results) {
            const PricingInput& run = test.first;
            double expected = test.second;
            double price = func(run.S, run.K, run.T, run.r, run.sigma, run.q, run.n, run.type);
            if (std::abs(price - expected) > 1e-5) {
                test_results.emplace_back(run, expected, price);
            }
        }
        return test_results;
    }

    SanityCheckResults run_single_all_sanity_checks_batch(BatchPricingFunction func) {
        SanityCheckResults test_results;
        std::vector<PricingInput> inputs;
        for (const auto& test : reference_function_results) {
            inputs.push_back(test.first);
        }
        std::vector<double> price(inputs.size());
        func(inputs, price);

        for (size_t i = 0; i < reference_function_results.size(); i++) {
            double expected = reference_function_results[i].second;
            if (std::abs(price[i] - expected) > 1e-5) {
                test_results.emplace_back(reference_function_results[i].first, expected, price[i]);
            }
        }
        return test_results;
    }
};
