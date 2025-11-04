#pragma once

#include <map>
#include <string>
#include <vector>

#include "benchmark_parameters.hpp"

class BenchmarkResult {
   public:
    Run run;
    std::string benchmark_parameters_name;
    std::map<int, double> execution_times;
    std::map<int, double> prices;
    std::string function_name;
    bool pass_sanity_check;

    BenchmarkResult(Run run, const std::string& benchmark_parameters_name,
                    const std::map<int, double>& execution_times, const std::string& function_name)
        : run(run),
          benchmark_parameters_name(benchmark_parameters_name),
          execution_times(execution_times),
          function_name(function_name),
          pass_sanity_check(true) {}

    BenchmarkResult(Run run, const std::string& benchmark_parameters_name,
                    const std::map<int, double>& execution_times, const std::string& function_name,
                    bool pass_sanity_check)
        : run(run),
          benchmark_parameters_name(benchmark_parameters_name),
          execution_times(execution_times),
          function_name(function_name),
          pass_sanity_check(pass_sanity_check) {}
};

inline std::string to_string(const BenchmarkResult& result) {
    std::string output =
        "Benchmark result for benchmark parameters: " + result.benchmark_parameters_name +
        ", function: " + result.function_name + "\n";
    output += to_string(result.run) + "\n";
    output += "Sanity check" + std::string(result.pass_sanity_check ? "passed" : "failed") + "\n";
    output += "Execution times (ms):\n";
    for (const auto& [n, time] : result.execution_times) {
        output += "  n=" + std::to_string(n) + ": " + std::to_string(time) + " ms\n";
    }
    return output;
}

std::vector<BenchmarkResult> benchmark(const std::string& filter_function_name,
                                       const std::string& benchmark_parameters,
                                       const std::string& reference_function_name,
                                       const bool skip_sanity_checks);
