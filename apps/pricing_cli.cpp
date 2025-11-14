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

std::pair<double, double> mean_and_std(const std::vector<double>& v) {
    double mean = 0., std = 0.;
    for (size_t i = 0; i < v.size(); i++) mean += v[i];
    mean /= v.size();

    for (size_t i = 0; i < v.size(); i++) std += (v[i] - mean) * (v[i] - mean);
    std /= (v.size() - 1);

    std = std::sqrt(std);
    return {mean, std};
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

void print_sanity_checks(const std::vector<BenchmarkResult>& results, bool skip_sanity_checks) {
    if (!skip_sanity_checks) {
        std::cout << "\n"
                  << rang::style::bold << "=== SANITY CHECKS SUMMARY ===" << rang::style::reset
                  << "\n\n";
        for (const auto& result : results) {
            for (const auto& [params, expected, price] : result.sanity_check_results) {
                std::cout << "Mismatch in " << result.function_name << " vs "
                          << result.reference_function_name << " (parameters: '" << params.name
                          << "'): expected " << expected << ", got " << price
                          << " (diff = " << std::abs(price - expected) << ")\n";
            }
        }
    }

    const int name_width = 80;
    const int status_width = 10;

    std::cout << std::left << std::setw(name_width) << "Function" << std::right
              << std::setw(status_width) << "Status"
              << "\n";
    std::cout << std::string(name_width + status_width, '-') << "\n";

    for (const auto& res : results) {
        bool pass = res.pass_sanity_check();
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

void print_benchmark_results_pprint(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n"
              << rang::style::bold << "=== BENCHMARK RESULTS ===" << rang::style::reset << "\n";
    std::cout << "Benchmark Parameters: " << to_string(results[0].run) << "\n\n";
    for (const auto& res : results) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << rang::style::bold;

        if (res.pass_sanity_check())
            std::cout << rang::fg::green;
        else
            std::cout << rang::fg::red;

        std::cout << "Function: " << res.function_name << rang::style::reset << "\n";
        std::cout << std::string(80, '-') << "\n";

        // Table header
        std::cout << std::left << std::setw(12) << "n" << std::right << std::setw(15) << "Time (ms)"
                  << std::right << std::setw(20) << "Output"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";

        // Table body
        for (const auto& [n_steps, time] : res.execution_times) {
            std::vector<double> price = res.prices.at(n_steps);

            if (time.size() > 1) {
                double mean_price = 0., mean_time = 0., std_price = 0., std_time = 0.;
                std::tie(mean_time, std_time) = mean_and_std(time);
                std::tie(mean_price, std_price) = mean_and_std(price);

                std::cout << std::left << std::setw(8) << n_steps << std::right << std::setw(15)
                          << std::fixed << std::setprecision(3) << mean_time << "±"
                          << std::setprecision(5) << std_time << std::right << std::setw(18)
                          << std::fixed << std::setprecision(6) << mean_price << "±"
                          << 3 * std_price << "\n";
            }
            if (time.size() == 1) {
                double mean_time = time[0], mean_price = price[0];

                std::cout << std::left << std::setw(12) << n_steps << std::right << std::setw(15)
                          << std::fixed << std::setprecision(3) << mean_time << std::right
                          << std::setw(20) << std::fixed << std::setprecision(6) << mean_price
                          << "\n";
            }
        }

        std::cout << std::string(80, '=') << "\n\n";
    }
}

std::pair<std::string, std::vector<std::string>> parse_hyperparams(const std::string& name) {
    std::string func_id = name;
    std::vector<std::string> hyperparams;

    size_t pos = name.find("_@");
    if (pos != std::string::npos) {
        func_id = name.substr(0, pos);
        std::string rest = name.substr(pos + 2);

        size_t start = 0;
        while ((pos = rest.find('_', start)) != std::string::npos) {
            hyperparams.push_back(rest.substr(start, pos - start));
            start = pos + 1;
        }
        if (start < rest.size()) hyperparams.push_back(rest.substr(start));
    }

    return std::make_pair(func_id, hyperparams);
}

void print_benchmark_results_json(const std::vector<BenchmarkResult>& results) {
    using json = nlohmann::json;
    json output;
    output["runs"] = json::array();

    if (!results.empty()) {
        auto run = results[0].run;
        output["benchmark_parameters"] = {{"S", run.S},
                                          {"K", run.K},
                                          {"T", run.T},
                                          {"r", run.r},
                                          {"sigma", run.sigma},
                                          {"q", run.q},
                                          {"nstart", run.nstart},
                                          {"nend", run.nend},
                                          {"nstep", run.nstep},
                                          {"nrepetition_at_step", run.nrepetition_at_step},
                                          {"type", to_string(run.type)}};
    }

    for (const auto& res : results) {
        auto [func_id, hparams] = parse_hyperparams(res.function_name);
        json::array_t n_vals, time_vals_mean, price_vals, hyper, time_vals_std, all_times;
        double std_time = 0., mean_time = 0.;
        for (const auto& [n_steps, time] : res.execution_times) {
            n_vals.push_back(n_steps);
            std::tie(mean_time, std_time) = mean_and_std(time);
            time_vals_mean.push_back(mean_time);
            time_vals_std.push_back(std_time);
            json::array_t time_at_n;
            for (auto ctime : time) {
                time_at_n.push_back(ctime);
            }
            all_times.push_back(time_at_n);
            price_vals.push_back(std::round(res.prices.at(n_steps)[0] * 1e6) / 1e6);
        }
        // json::array_t sanity_check_res =  res.sanity_check_results;
        json::array_t hyper_params;
        for (const auto& hparam : hparams) {
            hyper_params.push_back(hparam);
        }
        json::array_t sanity_checks;
        for (auto& [pricing_input, expected, price] : res.sanity_check_results) {
            json sanity_check;
            sanity_check["test_name"] = pricing_input.name;
            sanity_check["expected"] = expected;
            sanity_check["price"] = price;
            sanity_checks.push_back(sanity_check);
        }

        output["runs"].push_back(
            {{"id", res.function_name},
             {"function_id", func_id},
             {"do_pass_sanity_check", res.pass_sanity_check() ? "true" : "false"},
             {"sanity_check", sanity_checks},
             {"hyperparams", hyper_params},
             {"all_times", all_times},
             {"runs",
              {{"n", n_vals},
               {"time_ms_mean", time_vals_mean},
               {"time_ms_std", time_vals_std},
               {"price", price_vals}}}});
    }
    std::cout << output.dump(1, '\t') << std::endl;
}

int main(int argc, char** argv) {
    CLI::App app{"CLI for Option Pricing and Benchmarking"};

    // Pricing subcommand
    std::string option_type_str, exercise_type_str, pricing_method_str, backend_str,
        output_format_str;
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
            print_benchmark_results_pprint(results);
        } else if (output_format == OutputFormat::JSON) {
            print_benchmark_results_json(results);
        }

    } else if (*list_parameters_subcommand) {
        list_benchmark_parameters();
    }

    return 0;
}
