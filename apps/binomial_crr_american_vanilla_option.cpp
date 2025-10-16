#include <CLI/CLI.hpp>
#include <algorithm>
#include <benchmark.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include "constants.hpp"
#include "cpu/binomial_crr_american_vanilla_option_cpu.hpp"
#include "dataset.hpp"

int main(int argc, char** argv) {
  CLI::App app{"QuantOptions - CLI for American Option Pricing"};

  std::string type, function_name;
  double S, K, T, r, sigma, q;
  int n;

  auto single = app.add_subcommand("single", "Run a single pricing query");
  single->add_option("--type", type, "Option type (call|put)")
      ->default_val("call")
      ->check(CLI::IsMember({"call", "put"}));
  single->add_option("-S", S, "Spot price")->default_val(100.0);
  single->add_option("-K", K, "Strike price")->default_val(100.0);
  single->add_option("-T", T, "Maturity (years)")->default_val(0.5);
  single->add_option("-r", r, "Risk-free interest rate")->default_val(0.03);
  single->add_option("--sigma", sigma, "Volatility")->default_val(0.2);
  single->add_option("-q", q, "Dividend yield")->default_val(0.015);
  single->add_option("-n", n, "Number of binomial steps")->default_val(60);
  single->add_option("--function", function_name, "Function name")
      ->default_val("binomial_crr_american_vanilla_option_cpu");

  std::string filter_name;
  std::string dataset;

  auto bench = app.add_subcommand("benchmark", "Run benchmark on dataset");
  bench->add_option("--filter-by-name", filter_name, "Filter by benchmark name")->default_val("");
  bench->add_option("--dataset", dataset, "Dataset name (e.g., SP500)")->default_val("easy");

  auto list = app.add_subcommand("listdataset", "List available datasets and their config");

  try {
    app.require_subcommand(1);  // Must choose one of the three
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (*single) {
    OptionType type_opt = (type == "call") ? OptionType::Call : OptionType::Put;
    if (FUNCTIONS.find(function_name) == FUNCTIONS.end()) {
      std::cerr << "Function not found: " << function_name << "\n";
      return 1;
    }
    double price = FUNCTIONS[function_name](T, S, K, r, sigma, q, n, type_opt);
    printf("Option Price: %.4f\n", price);
  } else if (*bench) {
    benchmark(filter_name, dataset);
  } else if (*list) {
    list_datasets();
  }

  return 0;
}
