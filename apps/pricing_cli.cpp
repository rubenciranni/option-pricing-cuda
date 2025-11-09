#include <CLI/CLI.hpp>
#include <algorithm>
#include <benchmark.hpp>
#include <cstdlib>
#include <iostream>
#include <rang.hpp>
#include <string>
#include <unordered_map>

#include "benchmark_parameters.hpp"
#include "constants.hpp"
#include "models/vanilla_american_binomial.hpp"
#include "models/vanilla_european_binomial.hpp"

int main(int argc, char** argv) {
    CLI::App app{"CLI for Option Pricing and Benchmarking"};

    // Pricing subcommand
    std::string option_type_str, exercise_type_str, pricing_method_str, backend_str;
    double S, K, T, r, sigma, q;
    int n;

    auto price_subcommand = app.add_subcommand("price", "Run a single pricing query");
    price_subcommand->add_option("--type", option_type_str, "Option type (call|put)")
        ->default_val("call")
        ->check(CLI::IsMember({"call", "put"}));
    price_subcommand
        ->add_option("--exercise", exercise_type_str, "Exercise type (american|european)")
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

    // Benchmarking subcommand
    std::string filter_name;
    std::string benchmark_parameters;
    std::string reference_function_name;
    bool skip_sanity_checks;

    auto benchmark_subcommand =
        app.add_subcommand("benchmark", "Run benchmark on benchmark parameters");
    benchmark_subcommand->add_option("--filter-by-name", filter_name, "Filter functions by name")
        ->default_val("");
    benchmark_subcommand->add_option("--parameters", benchmark_parameters, "Parameters identifier")
        ->default_val("easy");
    benchmark_subcommand
        ->add_option("--reference-function", reference_function_name,
                     "Reference function name for sanity checks")
        ->default_val("vanilla_american_binomial_cpu_naive");
    benchmark_subcommand
        ->add_flag("--skip-sanity-checks", skip_sanity_checks, "Skip sanity checks.")
        ->default_val(false);
    skip_sanity_checks = false;

    // List parameters subcommand
    auto list_parameters_subcommand = app.add_subcommand(
        "benchmark-parameters", "List available benchmark parameters and their config");

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
    } else if (*benchmark_subcommand) {
        if (!skip_sanity_checks) {
            std::cout << rang::style::bold << rang::fg::red << "=== SANITY CHECKS LOGS ===\n\n"
                      << rang::style::reset;
        }

        // Run benchmark
        auto results = benchmark(filter_name, benchmark_parameters, reference_function_name,
                                 skip_sanity_checks);

        // ==========================================
        // Sanity Check Summary
        // ==========================================
        if (!skip_sanity_checks) {
            std::cout << "\n"
                      << rang::style::bold << "=== SANITY CHECKS SUMMARY ===" << rang::style::reset
                      << "\n\n";

            const int name_width = 65;
            const int status_width = 10;

            std::cout << std::left << std::setw(name_width) << "Function" << std::right
                      << std::setw(status_width) << "Status"
                      << "\n";
            std::cout << std::string(name_width + status_width, '-') << "\n";

            for (const auto& res : results) {
                bool pass = res.pass_sanity_check;
                std::string status = pass ? "✅" : "❌";

                if (pass)
                    std::cout << rang::fg::green;
                else
                    std::cout << rang::fg::red;

                std::cout << std::left << std::setw(name_width) << res.function_name << std::right
                          << std::setw(status_width) << status << rang::fg::reset << "\n";
            }

            std::cout << std::string(name_width + status_width, '-') << "\n\n";
        }

        // ==========================================
        // Benchmark Parameters
        // ==========================================
        if (results.empty()) {
            std::cout << rang::fg::yellow
                      << "No functions matched the given filter: " << filter_name << rang::fg::reset
                      << "\n";
            return 0;
        }

        // ==========================================
        // Results Per Function
        // ==========================================
        std::cout << "\n"
                  << rang::style::bold << "=== BENCHMARK RESULTS ===" << rang::style::reset << "\n";
        std::cout << "Benchmark Parameters: " << to_string(results[0].run) << "\n\n";
        for (const auto& res : results) {
            std::cout << std::string(80, '=') << "\n";
            std::cout << rang::style::bold;

            if (res.pass_sanity_check)
                std::cout << rang::fg::green;
            else
                std::cout << rang::fg::red;

            std::cout << "Function: " << res.function_name << rang::style::reset << "\n";
            std::cout << std::string(80, '-') << "\n";

            // Table header
            std::cout << std::left << std::setw(12) << "n" << std::right << std::setw(15)
                      << "Time (ms)" << std::right << std::setw(20) << "Output"
                      << "\n";
            std::cout << std::string(50, '-') << "\n";

            // Table body
            for (const auto& [n_steps, time] : res.execution_times) {
                double price = res.prices.at(n_steps);
                std::cout << std::left << std::setw(12) << n_steps << std::right << std::setw(15)
                          << std::fixed << std::setprecision(3) << time << std::right
                          << std::setw(20) << std::fixed << std::setprecision(6) << price << "\n";
            }

            std::cout << std::string(80, '=') << "\n\n";
        }

    } else if (*list_parameters_subcommand) {
        list_benchmark_parameters();
    }
    return 0;
}
