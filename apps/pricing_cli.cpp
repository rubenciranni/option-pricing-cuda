#include "include/pricing_cli.hpp"

#include <CLI/CLI.hpp>
#include <algorithm>
#include <benchmark.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <rang.hpp>
#include <streambuf>
#include <string>
#include <unordered_map>

#include "benchmark_parameters.hpp"
#include "constants.hpp"
#include "models/vanilla_american_binomial.hpp"
#include "models/vanilla_european_binomial.hpp"

#define DEBUG 1

int main(int argc, char** argv) {
    CLI::App app{"CLI for Option Pricing and Benchmarking"};

    // Pricing subcommand
    std::string option_type_str, exercise_type_str, pricing_method_str, backend_str,
        output_format_str;
    double S, K, T, r, sigma, q;
    int n, random_runs;

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
    int skip_sanity_checks;

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
    benchmark_subcommand
        ->add_option("--output-format", output_format_str, "Output Format (pprint|json)")
        ->default_val("pprint")
        ->check(CLI::IsMember({"pprint", "json"}));

    // List parameters subcommand
    auto list_parameters_subcommand = app.add_subcommand(
        "benchmark-parameters", "List available benchmark parameters and their config");

    auto random_benchmark_subcommand =
        app.add_subcommand("random-benchmark", "Run benchmark on randomly generated parameters");
    random_benchmark_subcommand
        ->add_option("--filter-by-name", filter_name, "Filter functions by name")
        ->default_val("");

    random_benchmark_subcommand
        ->add_option("--reference-function", reference_function_name,
                     "Reference function name for sanity checks")
        ->default_val("vanilla_american_binomial_cuda_stprcmp");
    random_benchmark_subcommand
        ->add_flag("--skip-sanity-checks", skip_sanity_checks, "Skip sanity checks.")
        ->default_val(false);
    random_benchmark_subcommand
        ->add_option("--output-format", output_format_str, "Output Format (pprint|json)")
        ->default_val("pprint")
        ->check(CLI::IsMember({"pprint", "json"}));
    random_benchmark_subcommand
        ->add_option("--n-random-runs", random_runs, "Number of random benchmark runs to perform")
        ->default_val(5);

    auto check_occupancy_subcommand =
        app.add_subcommand("check-occupancy", "Check occupancy of CUDA implementations");

    auto benchmark_random_thougput_subcommand =
        app.add_subcommand("benchmark-random-throughput",
                           "Run benchmark on randomly generated parameters for throughput");
    benchmark_random_thougput_subcommand
        ->add_option("--filter-by-name", filter_name, "Filter functions by name")
        ->default_val("");
    benchmark_random_thougput_subcommand
        ->add_option("--output-format", output_format_str, "Output Format (pprint|json)")
        ->default_val("pprint")
        ->check(CLI::IsMember({"pprint", "json"}));

    auto batch_random_benchmark_subcommand = app.add_subcommand(
        "batch-random-benchmark", "Run benchmark on randomly generated parameters");
    batch_random_benchmark_subcommand
        ->add_option("--filter-by-name", filter_name, "Filter functions by name")
        ->default_val("");

    batch_random_benchmark_subcommand
        ->add_option("--reference-function", reference_function_name,
                     "Reference function name for sanity checks")
        ->default_val("vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds");
    batch_random_benchmark_subcommand
        ->add_flag("--skip-sanity-checks", skip_sanity_checks, "Skip sanity checks.")
        ->default_val(false);
    batch_random_benchmark_subcommand
        ->add_option("--output-format", output_format_str, "Output Format (pprint|json)")
        ->default_val("pprint")
        ->check(CLI::IsMember({"pprint", "json"}));
    batch_random_benchmark_subcommand
        ->add_option("--n-random-runs", random_runs, "Number of random benchmark runs to perform")
        ->default_val(5);
    batch_random_benchmark_subcommand->add_option("-n", n, "Number of step 'n' in each random run")
        ->default_val(1000);
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
        OutputFormat output_format = output_format_from_string(output_format_str);

        auto results = benchmark(filter_name, benchmark_parameters, reference_function_name,
                                 skip_sanity_checks);

        if (results.empty()) {
            std::cout << rang::fg::yellow
                      << "No functions matched the given filter: " << filter_name << rang::fg::reset
                      << "\n";
            return 0;
        }
        if (output_format == OutputFormat::PPRINT) {
            print_sanity_checks(results, skip_sanity_checks);
            print_benchmark_results(results);
        } else if (output_format == OutputFormat::JSON) {
            nlohmann::json output = dump_benchmark_results_json(results);
            std::cout << output.dump(1, '\t') << std::endl;
        }

    } else if (*list_parameters_subcommand) {
        list_benchmark_parameters();
    } else if (*random_benchmark_subcommand) {
        OutputFormat output_format = output_format_from_string(output_format_str);

        auto results =
            random_benchmark(filter_name, reference_function_name, random_runs, skip_sanity_checks);

        if (results.empty()) {
            std::cout << rang::fg::yellow
                      << "No functions matched the given filter: " << filter_name << rang::fg::reset
                      << "\n";
            return 0;
        }
        if (output_format == OutputFormat::PPRINT) {
            print_sanity_checks(results[0], skip_sanity_checks);
            for (const auto& res_per_run : results) {
                print_benchmark_results(res_per_run);
            }
        } else if (output_format == OutputFormat::JSON) {
            nlohmann::json output = nlohmann::json::array();

            for (const auto& res_per_run : results) {
                output.push_back(dump_benchmark_results_json(res_per_run));
            }
            std::cout << output.dump(1, '\t') << std::endl;
        }
    } else if (*batch_random_benchmark_subcommand) {
        auto results = batch_random_benchmark(filter_name, reference_function_name, random_runs, n,
                                              skip_sanity_checks);

        OutputFormat output_format = output_format_from_string(output_format_str);
        if (output_format == OutputFormat::PPRINT) {
            print_batch_benchmark_result(results, skip_sanity_checks);
        } else if (output_format == OutputFormat::JSON) {
            auto output = dump_batch_benchmark_results_json(results);
            std::cout << output.dump(1, '\t') << std::endl;
        }
    } else if (*check_occupancy_subcommand) {
        check_occupancy_all_cuda_functions();

    } else if (*benchmark_random_thougput_subcommand) {
        nlohmann::json::array_t output;
        for (auto random_runs : std::vector<int>{1,2,4,8,16,32,64,128,256,512,1024}) {
            for (auto n : std::vector<int>{64,128,256,512,1024,2048,4096,8192,1<<14,1<<15,1<<16,1<<17}) {
                auto results = batch_random_benchmark(filter_name, reference_function_name,
                                                      random_runs, n, skip_sanity_checks);
                OutputFormat output_format = output_format_from_string(output_format_str);
                if (output_format == OutputFormat::PPRINT) {
                    print_batch_benchmark_result(results, skip_sanity_checks);
                } else if (output_format == OutputFormat::JSON) {
                    auto batch_output = dump_batch_benchmark_results_json(results);
                    output.insert(output.end(), batch_output.begin(), batch_output.end());
                }
            }
        }
        std::cout << output << std::endl;
    }

    return 0;
}
