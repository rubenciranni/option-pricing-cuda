#include "sanity_checks.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"

std::vector<SingleRun> TESTS = {SingleRun(100, 100, 0.5, 0.03, 0.2, 0.015, 60, OptionType::Put)};

bool run_single_sanity_check(PricingFunction func, const std::pair<double, SingleRun>& test) {
  double expected = test.first;
  const SingleRun& run = test.second;
  double price = func(run.S, run.K, run.T, run.r, run.sigma, run.q, run.n, run.type);
  if (std::abs(price - expected) > 1e-2) {
    return false;
  }
  return true;
}

bool run_multiple_sanity_checks(PricingFunction func,
                                const std::vector<std::pair<double, SingleRun>>& tests) {
  for (const auto& test : tests) {
    if (!run_single_sanity_check(func, test)) {
      return false;
    }
  }
  return true;
}

bool run_sanity_checks_by_function(const PricingFunction fun) {
  std::vector<std::pair<double, SingleRun>> TESTS_WITH_RESULTS = {};
  for (const auto& run : TESTS) {
    TESTS_WITH_RESULTS.push_back({vanilla_american_binomial_cpu(run.S, run.K, run.T, run.r,
                                                                run.sigma, run.q, run.n, run.type),
                                  run});
  }
  return run_multiple_sanity_checks(fun, TESTS_WITH_RESULTS);
}
