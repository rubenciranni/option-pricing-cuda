#pragma once

#include <chrono>
#include <string>

#include "benchmark_parameters.hpp"
#include "functions_version.hpp"
#include "sanity_checks.hpp"

#include <vector>

class Result {
 public:
  Run run;
  std::string benchmark_parameters_name;
  std::map<int, double> execution_times;
  std::string function_name;
  bool pass_sanity_check;

  Result(Run run, const std::string& benchmark_parameters_name,
         const std::map<int, double>& execution_times, const std::string& function_name)
      : run(run),
        benchmark_parameters_name(benchmark_parameters_name),
        execution_times(execution_times),
        function_name(function_name),
        pass_sanity_check(true) {}

  Result(Run run, const std::string& benchmark_parameters_name,
         const std::map<int, double>& execution_times, const std::string& function_name,
         bool pass_sanity_check)
      : run(run),
        benchmark_parameters_name(benchmark_parameters_name),
        execution_times(execution_times),
        function_name(function_name),
        pass_sanity_check(pass_sanity_check) {}
};

inline std::string to_string(const Result& result) {
  std::string output = "Result for benchmark_parameters: " + result.benchmark_parameters_name +
                       ", function: " + result.function_name + "\n";
  output += to_string(result.run) + "\n";
  output +=
      "Sanity check passed: " + std::string(result.pass_sanity_check ? "true" : "false") + "\n";
  output += "Execution times (ms):\n";
  for (const auto& [n, time] : result.execution_times) {
    output += "  n=" + std::to_string(n) + ": " + std::to_string(time) + " ms\n";
  }
  return output;
}

std::vector<Result> benchmark(const std::string& filter_function_name,
                              const std::string& benchmark_parameters, const bool no_verify);
