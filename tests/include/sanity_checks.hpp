#pragma once
#include <vector>

#include "dataset.hpp"
#include "functions_version.hpp"

bool run_single_sanity_check(PricingFunction func, const std::pair<double, SingleRun>& test);

bool run_multiple_sanity_checks(PricingFunction func,
                                const std::vector<std::pair<double, SingleRun>>& tests);

bool run_sanity_checks_by_function(const PricingFunction fun);

extern std::vector<SingleRun> TESTS;
