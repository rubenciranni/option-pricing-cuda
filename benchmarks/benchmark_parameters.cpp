#include "benchmark_parameters.hpp"

std::map<std::string, Run> BENCHMARK_PARAMETERS = {
    {"debug", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 20, 20, 20, OptionType::Put)},
    {"easy", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 1000, 2000, 1000, OptionType::Put)},
    {"hard", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 20000, 10000, OptionType::Put)},
};

void list_benchmark_parameters() {
  std::cout << "Available benchmark parameters identifiers:\n";
  for (const auto& [name, run] : BENCHMARK_PARAMETERS) {
    std::cout << "  - " << name << ": ";
    std::cout << to_string(run) << "\n";
  }
}
