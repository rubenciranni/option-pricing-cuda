#include <CLI/CLI.hpp>
#include <algorithm>
#include <benchmark.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include "benchmark_parameters.hpp"
#include "constants.hpp"
#include "models/vanilla_american_binomial.hpp"
#include "models/vanilla_european_binomial.hpp"

int main(int argc, char** argv) {
  CLI::App app{"CLI for Option Pricing and Benchmarking"};

  std::string option_type_str, exercise_type_str, pricing_method_str, backend_str;
  double S, K, T, r, sigma, q;
  int n;

  auto price_subcommand = app.add_subcommand("price", "Run a single pricing query");
  price_subcommand->add_option("--type", option_type_str, "Option type (call|put)")
      ->default_val("call")
      ->check(CLI::IsMember({"call", "put"}));
  price_subcommand->add_option("--exercise", exercise_type_str, "Exercise type (american|european)")
      ->default_val("american")
      ->check(CLI::IsMember({"american", "european"}));
  price_subcommand->add_option("--method", pricing_method_str, "Pricing method (binomial)")
      ->default_val("binomial");
  price_subcommand->add_option("-S", S, "Spot price")->default_val(100.0);
  price_subcommand->add_option("-K", K, "Strike price")->default_val(100.0);
  price_subcommand->add_option("-T", T, "Maturity (years)")->default_val(0.5);
  price_subcommand->add_option("-r", r, "Risk-free interest rate")->default_val(0.03);
  price_subcommand->add_option("--sigma", sigma, "Volatility")->default_val(0.2);
  price_subcommand->add_option("-q", q, "Dividend yield")->default_val(0.015);
  price_subcommand->add_option("-n", n, "Number of binomial steps")->default_val(60);
  price_subcommand->add_option("--backend", backend_str, "Backend (cpu|openmp|cuda)")
      ->default_val("cpu")
      ->check(CLI::IsMember({"cpu", "openmp", "cuda"}));
  std::string filter_name;
  std::string benchmark_parameters;

  auto bench = app.add_subcommand("benchmark", "Run benchmark on benchmark_parameters");
  bench->add_option("--filter-by-name", filter_name, "Filter by benchmark name")->default_val("");
  bench->add_option("--parameters", benchmark_parameters, "Parameters identifier")
      ->default_val("easy");

  auto list =
      app.add_subcommand("parameters", "List available benchmark parameters and their config");

  try {
    app.require_subcommand(1);
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (*price_subcommand) {
    OptionType option_type = option_type_from_string(option_type_str);
    Backend backend = backend_from_string(backend_str);
    ExerciseType exercise_type = exercise_type_from_string(exercise_type_str);
    PricingMethod pricing_method = pricing_method_from_string(pricing_method_str);

    double price = -1.;
    if (pricing_method == PricingMethod::Binomial) {
      if (exercise_type == ExerciseType::American) {
        price = vanilla_american_binomial(S, K, T, r, sigma, q, n, option_type, backend);
      } else if (exercise_type == ExerciseType::European) {
        price = vanilla_european_binomial(S, K, T, r, sigma, q, n, option_type, backend);
      }
    }

    printf("Option Price: %.4f\n", price);
  } else if (*bench) {
    benchmark(filter_name, benchmark_parameters);
  } else if (*list) {
    list_benchmark_parameterss();
  }

  return 0;
}
