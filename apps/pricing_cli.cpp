#include <CLI/CLI.hpp>
#include <algorithm>
#include <benchmark.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <rang.hpp>
#include <streambuf>
#include <string>
#include <unordered_map>

#include "benchmark_parameters.hpp"
#include "constants.hpp"
#include "models/vanilla_american_binomial.hpp"
#include "models/vanilla_european_binomial.hpp"

class JsonStreamBuf : public std::streambuf {
    std::streambuf* original;
    std::string buffer;
    bool first = true;
    int json_depth;

   protected:
    // Called whenever a character is inserted
    int overflow(int c) override {
        if (c != EOF) {
            buffer += static_cast<char>(c);
            if (c == '\n') {
                // flush a full line
                flushBuffer();
            }
        }
        return c;
    }

    int sync() override {
        flushBuffer();
        return 0;
    }

    void flushBuffer() {
        if (!buffer.empty()) {
            // Example: wrap each line in quotes
            std::string formatted = "";
            if (first)
                first = false;
            else
                formatted += ",\n";
            formatted +=
                std::string(json_depth, '\t') + "{\"message\": \"" + escapeJson(buffer) + "\"}";
            original->sputn(formatted.c_str(), formatted.size());
            buffer.clear();
        }
    }

    static std::string escapeJson(const std::string& s) {
        std::string out;
        for (char c : s) {
            switch (c) {
                case '\"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    out += c;
            }
        }
        return out;
    }

   public:
    explicit JsonStreamBuf(std::streambuf* buf, int json_depth)
        : original(buf), json_depth(json_depth) {}
};

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
        // ==========================================
        // Results Per Function
        // ==========================================
        OutputFormat output_format = output_format_from_string(output_format_str);
        if (output_format == OutputFormat::PPRINT) {
            // ==========================================
            // Sanity Check Summary
            // ==========================================
            if (!skip_sanity_checks) {
                std::cout << rang::style::bold << rang::fg::red << "=== SANITY CHECKS LOGS ===\n\n"
                          << rang::style::reset;
            }

            // Run benchmark
            auto results = benchmark(filter_name, benchmark_parameters, reference_function_name,
                                     skip_sanity_checks);

            // ==========================================
            // Benchmark Parameters
            // ==========================================
            if (results.empty()) {
                std::cout << rang::fg::yellow
                          << "No functions matched the given filter: " << filter_name
                          << rang::fg::reset << "\n";
                return 0;
            }

            if (!skip_sanity_checks) {
                std::cout << "\n"
                          << rang::style::bold
                          << "=== SANITY CHECKS SUMMARY ===" << rang::style::reset << "\n\n";

                const int name_width = 80;
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

                    std::cout << std::left << std::setw(name_width) << res.function_name
                              << std::right << std::setw(status_width) << status << rang::fg::reset
                              << "\n";
                }

                std::cout << std::string(name_width + status_width, '-') << "\n\n";
            }

            // ==========================================
            // Benchmark Results
            // ==========================================
            std::cout << "\n"
                      << rang::style::bold << "=== BENCHMARK RESULTS ===" << rang::style::reset
                      << "\n";
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
                std::cout << std::string(70, '-') << "\n";

                // Table body
                for (const auto& [n_steps, time] : res.execution_times) {
                    std::vector<double> price = res.prices.at(n_steps);

                    if (time.size() > 1) {
                        double mean_price = 0., mean_time = 0., std_price = 0., std_time = 0.;
                        std::tie(mean_time, std_time) = mean_and_std(time);
                        std::tie(mean_price, std_price) = mean_and_std(price);

                        std::cout << std::left << std::setw(8) << n_steps << std::right
                                  << std::setw(15) << std::fixed << std::setprecision(3)
                                  << mean_time << "±" << std::setprecision(5) << std_time
                                  << std::right << std::setw(18) << std::fixed
                                  << std::setprecision(6) << mean_price << "±" << 3 * std_price
                                  << "\n";
                    }
                    if (time.size() == 1) {
                        double mean_time = time[0], mean_price = price[0];

                        std::cout << std::left << std::setw(12) << n_steps << std::right
                                  << std::setw(15) << std::fixed << std::setprecision(3)
                                  << mean_time << std::right << std::setw(20) << std::fixed
                                  << std::setprecision(6) << mean_price << "\n";
                    }
                }

                std::cout << std::string(80, '=') << "\n\n";
            }

        } else if (*list_parameters_subcommand) {
            list_benchmark_parameters();
        }
        if (output_format == OutputFormat::JSON) {
            int json_depth = 1;
            std::string json_output = "{\n";
            json_output += std::string(json_depth, '\t') + "\"errors\": [";
            json_depth++;
            std::cout << json_output << std::endl;
            json_output = "";

            std::streambuf* originalBuf = std::cout.rdbuf();
            JsonStreamBuf jsonBuf(std::cout.rdbuf(), json_depth);
            std::cout.rdbuf(&jsonBuf);
            auto results = benchmark(filter_name, benchmark_parameters, reference_function_name,
                                     skip_sanity_checks);
            std::cout.rdbuf(originalBuf);

            json_depth -= 1;
            json_output += '\n' + std::string(json_depth, '\t') + "],\n";
            json_output += std::string(json_depth, '\t') + "\"runs\": [\n";
            json_depth++;

            bool first = true;
            for (const auto& res : results) {
                if (first == true)
                    first = false;
                else
                    json_output += ",\n";
                json_output += std::string(json_depth, '\t') + "{\n";
                json_depth++;

                std::string json_hyperparam_delimiter = "_@", function_id = res.function_name,
                            hyperparam_id = "null", param_1 = "null", param_2 = "null",
                            param_3 = "null", param_4 = "null", param_5 = "null";
                if (res.function_name.find(json_hyperparam_delimiter) != std::string::npos) {
                    int json_delimiter_start = 0,
                        json_delimiter_end = res.function_name.find(json_hyperparam_delimiter);
                    function_id = res.function_name.substr(
                        json_delimiter_start, json_delimiter_end - json_delimiter_start);
                    json_delimiter_start += json_delimiter_end + json_hyperparam_delimiter.length();

                    json_delimiter_end = res.function_name.find('_', json_delimiter_start);
                    hyperparam_id = res.function_name.substr(
                        json_delimiter_start, json_delimiter_end - json_delimiter_start);
                    json_delimiter_start = json_delimiter_end + 1;

                    json_delimiter_end = res.function_name.find('_', json_delimiter_start);
                    param_1 = res.function_name.substr(json_delimiter_start,
                                                       json_delimiter_end - json_delimiter_start);
                    json_delimiter_start = json_delimiter_end + 1;

                    json_delimiter_end = res.function_name.find('_', json_delimiter_start);
                    param_2 = res.function_name.substr(json_delimiter_start,
                                                       json_delimiter_end - json_delimiter_start);
                    json_delimiter_start = json_delimiter_end + 1;

                    json_delimiter_end = res.function_name.find('_', json_delimiter_start);
                    param_3 = res.function_name.substr(json_delimiter_start,
                                                       json_delimiter_end - json_delimiter_start);
                    json_delimiter_start = json_delimiter_end + 1;

                    json_delimiter_end = res.function_name.find('_', json_delimiter_start);
                    param_4 = res.function_name.substr(json_delimiter_start,
                                                       json_delimiter_end - json_delimiter_start);
                    json_delimiter_start = json_delimiter_end + 1;

                    param_5 = res.function_name.substr(
                        json_delimiter_start, res.function_name.length() - json_delimiter_start);
                }
                json_output +=
                    std::string(json_depth, '\t') + "\"id\": \"" + res.function_name + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"function_id\": \"" + function_id + "\",\n";
                json_output += std::string(json_depth, '\t') + "\"do_pass_sanity_check\": \"" +
                               (res.pass_sanity_check ? "true" : "false") + "\",\n";
                json_output += std::string(json_depth, '\t') + "\"hyperparam_id\": \"" +
                               hyperparam_id + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"hyperparam_1\": \"" + param_1 + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"hyperparam_2\": \"" + param_2 + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"hyperparam_3\": \"" + param_3 + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"hyperparam_4\": \"" + param_4 + "\",\n";
                json_output +=
                    std::string(json_depth, '\t') + "\"hyperparam_5\": \"" + param_5 + "\",\n";
                json_output += std::string(json_depth, '\t') + "\"runs\": {\n";
                json_depth += 1;

                std::string json_nsteps_string = "", json_time_string = "", json_price_string = "";
                for (const auto& [n_steps, time] : res.execution_times) {
                    std::vector<double> price = res.prices.at(n_steps);
                    json_nsteps_string += (", " + std::to_string(n_steps));
                    json_time_string += (", " + to_string_with_precision(time, 3));
                    json_price_string += (", " + to_string_with_precision(price, 6));
                }

                json_output += std::string(json_depth, '\t') + "\"n\": [" +
                               json_nsteps_string.substr(2, json_nsteps_string.length() - 2) +
                               "],\n";
                json_output += std::string(json_depth, '\t') + "\"time_ms\": [" +
                               json_time_string.substr(2, json_time_string.length() - 2) + "],\n";
                json_output += std::string(json_depth, '\t') + "\"price\": [" +
                               json_price_string.substr(2, json_price_string.length() - 2) + "]\n";
                json_depth -= 1;
                json_output += std::string(json_depth, '\t') + "}\n";
                json_depth -= 1;
                json_output += std::string(json_depth, '\t') + "}";
                std::cout << json_output;
                json_output = "";
            }
            json_depth -= 1;
            json_output += '\n' + std::string(json_depth, '\t') + "]\n";
            json_depth -= 1;
            json_output += std::string(json_depth, '\t') + "}\n";
            std::cout << json_output << std::endl;
        }
    }

    return 0;
}
