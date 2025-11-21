#pragma once

#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "benchmark.hpp"

// Utility helpers
std::pair<double, double> mean_and_std(const std::vector<double>& v);

template <typename T>
inline std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Printing helpers used by the CLI
void print_sanity_checks(const std::vector<BenchmarkResult>& results, bool skip_sanity_checks);
void print_table(const std::vector<int>& max_width, const std::vector<std::vector<std::string>>& table);
void print_benchmark_results_pprint(const std::vector<BenchmarkResult>& results);

nlohmann::json dump_benchmark_results_json(const std::vector<BenchmarkResult>& results);

// Parsing helpers
std::pair<std::string, std::vector<std::string>> parse_hyperparams(const std::string& name);

