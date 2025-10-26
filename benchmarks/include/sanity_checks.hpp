#pragma once
#include <vector>

#include "benchmark_parameters.hpp"
#include "functions_version.hpp"

extern std::vector<SingleRun> TESTS;

class TestFunctionSanityChecks {
 public:
  std::string truth_fun_name;
  PricingFunction truth_fun;
  std::vector<std::pair<double, SingleRun>> TESTS_WITH_RESULTS;
  bool with_logging = true;

  TestFunctionSanityChecks(std::string name, PricingFunction function)
      : truth_fun_name(name), truth_fun(function) {
    for (const auto& run : TESTS) {
      TESTS_WITH_RESULTS.push_back(
          {function(run.S, run.K, run.T, run.r, run.sigma, run.q, run.n, run.type), run});
    }
  }

  bool run_single_all_sanity_checks(std::string fun_name, PricingFunction func) {
    bool all_passed = true;
    for (const auto& test : TESTS_WITH_RESULTS) {
      double expected = test.first;
      const SingleRun& run = test.second;
      double price = func(run.S, run.K, run.T, run.r, run.sigma, run.q, run.n, run.type);
      if (std::abs(price - expected) > 1e-5) {
        if (with_logging) {
          std::cout << "function " << fun_name << " has an error compare to  " << truth_fun_name
                    << " (dataset name '" << run.name << "') expected: " << expected
                    << " got: " << price << std::endl;
        }
        all_passed = false;
      }
    }
    return all_passed;
  }
};
