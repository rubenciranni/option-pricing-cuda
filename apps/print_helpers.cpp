#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <rang.hpp>
#include <sstream>

#include "benchmark_parameters.hpp"
#include "constants.hpp"
#include "include/pricing_cli.hpp"
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

void print_table(const std::vector<int>& max_width,
                 const std::vector<std::vector<std::string>>& table) {
    if (table.empty()) return;

    size_t cols = max_width.size();
    size_t total_width = 0;
    for (auto w : max_width) total_width += w;
    // account for a single space separator between columns
    if (cols > 1) total_width += (cols - 1);

    std::cout << std::string(total_width, '=') << "\n";

    // Header is the first row
    const auto& header = table[0];
    std::cout << rang::style::bold;
    if (!header.empty()) {
        // print first column left-aligned, rest right-aligned with a separating space
        std::cout << std::left << std::setw(max_width[0]) << header[0];
        for (size_t c = 1; c < cols; ++c) {
            std::string cell = (c < header.size()) ? header[c] : std::string("");
            std::cout << ' ' << std::right << std::setw(max_width[c]) << cell;
        }
    }
    std::cout << rang::style::reset << "\n";
    std::cout << std::string(total_width, '=') << "\n";

    // print remaining rows
    for (size_t r = 1; r < table.size(); ++r) {
        const auto& row = table[r];

        if (!row.empty()) {
            std::cout << std::left << std::setw(max_width[0]) << row[0];
            for (size_t c = 1; c < cols; ++c) {
                std::string cell = (c < row.size()) ? row[c] : std::string("-");
                if (cell.length() > static_cast<size_t>(max_width[c])) {
                    cell = cell.substr(0, max_width[c] - 3) + "...";
                }
                std::cout << ' ' << std::right << std::setw(max_width[c]) << cell;
            }
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '=') << "\n\n";
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

    const int status_width = 10;

    size_t max_name_len = std::string("Function").size();
    for (const auto& res : results) {
        max_name_len = std::max(max_name_len, res.function_name.size());
    }

    std::vector<int> max_width = {static_cast<int>(max_name_len), status_width};

    std::vector<std::vector<std::string>> table;
    table.push_back({"Function", "Status"});

    for (const auto& res : results) {
        std::string status = res.pass_sanity_check() ? "✅" : "❌";
        table.push_back({res.function_name, status});
    }

    print_table(max_width, table);
}

void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n"
              << rang::style::bold << "=== BENCHMARK RESULTS ===" << rang::style::reset << "\n";
    if (results.empty()) {
        std::cout << rang::fg::yellow << "No benchmark results to display." << rang::fg::reset
                  << "\n\n";
        return;
    }

    std::cout << "Benchmark Parameters: " << to_string(results[0].run) << "\n\n";

    int max_functional_name_size = 85;
    int max_width_step_size = 15;

    // build column widths: first column + one per step
    std::vector<int> max_width;
    max_width.push_back(max_functional_name_size);
    std::fill_n(std::back_inserter(max_width), results[0].execution_times.size(),
                max_width_step_size);

    std::vector<std::vector<std::string>> table_times;
    std::vector<std::vector<std::string>> table_prices;
    std::vector<std::string> header;
    header.push_back("Function");
    for (const auto& [n_steps, _] : results[0].execution_times) {
        header.push_back("n=" + std::to_string(n_steps));
    }
    table_times.push_back(header);
    table_prices.push_back(header);

    // Helper to find times vector for a given n_steps in a run
    auto find_times = [](const BenchmarkResult& run, int n) -> const std::vector<double>& {
        for (const auto& p : run.execution_times) {
            if (p.first == n) return p.second;
        }
        throw std::runtime_error("No execution times found for n_steps=" + std::to_string(n));
    };

    // Helper to find prices vector for a given n_steps in a run
    auto find_prices = [](const BenchmarkResult& run, int n) -> const std::vector<double>& {
        for (const auto& p : run.prices) {
            if (p.first == n) return p.second;
        }
        throw std::runtime_error("No execution times found for n_steps=" + std::to_string(n));
    };

    for (const auto& res : results) {
        std::string function_title = res.function_name;
        std::vector<std::string> row_times;
        std::vector<std::string> row_prices;
        // prepend a simple pass/fail marker to title for quick visibility
        std::string marker = res.pass_sanity_check() ? "✅ " : "❌ ";
        row_times.push_back(marker + function_title);
        row_prices.push_back(marker + function_title);

        for (const auto& [n_steps, _] : results[0].execution_times) {
            const std::vector<double>& times = find_times(res, n_steps);
            const std::vector<double>& prices = find_prices(res, n_steps);
            double mean_time = 0.0, std_time = 0.0;
            double mean_price = 0.0, std_price = 0.0;
            if (!times.empty()) {
                if (times.size() == 1) {
                    mean_time = times[0];
                    std_time = 0.0;
                } else {
                    std::tie(mean_time, std_time) = mean_and_std(times);
                }
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3) << mean_time;
                if (std_time > 0.0) {
                    oss << "±" << std::fixed << std::setprecision(3) << std_time;
                }
                row_times.push_back(oss.str());
            } else {
                row_times.push_back("-");
            }
            if (!prices.empty()) {
                if (prices.size() == 1) {
                    mean_price = prices[0];
                    std_price = 0.0;
                } else {
                    std::tie(mean_price, std_price) = mean_and_std(prices);
                }
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(5) << mean_price;
                if (std_price > 0.0) {
                    oss << "±" << std::fixed << std::setprecision(5) << std_price;
                }
                row_prices.push_back(oss.str());
            } else {
                row_prices.push_back("-");
            }
        }

        table_times.push_back(row_times);
        table_prices.push_back(row_prices);
    }
    std::cout << "Times\n";
    print_table(max_width, table_times);
    std::cout << "Outputs\n";
    print_table(max_width, table_prices);
}

void print_batch_benchmark_result(const std::vector<BatchBenchmarkResult>& results) {
    std::cout << "\n"
              << rang::style::bold << "=== BENCHMARK RESULTS ===" << rang::style::reset << "\n";
    if (results.empty()) {
        std::cout << rang::fg::yellow << "No benchmark results to display." << rang::fg::reset
                  << "\n\n";
        return;
    }

    int max_functional_name_size = 85;
    int max_width_step_size = 15;

    // build column widths: first column + one per step
    std::vector<int> max_width;
    max_width.push_back(max_functional_name_size);
    max_width.push_back(max_width_step_size);

    std::vector<std::vector<std::string>> table;
    std::vector<std::string> header;
    header.push_back("Function");
    header.push_back("Time (ms)");
    table.push_back(header);

    for (size_t i = 0; i < results.size(); ++i) {
        std::vector<std::string> row;
        const auto& res = results[i];
        std::string function_title = res.function_name;
        std::string marker = res.pass_sanity_check() ? "✅ " : "❌ ";

        row.push_back(marker + function_title);
        double mean, std;
        std::tie(mean, std) = mean_and_std(res.execution_times);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << mean;
        if (std > 0.0) {
            oss << "±" << std::fixed << std::setprecision(3) << std;
        }
        row.push_back(oss.str());
        table.push_back(row);
    }
    print_table(max_width, table);
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

nlohmann::json dump_run_json(const Run& run) {
    return nlohmann::json{{"S", run.S},
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

nlohmann::json dump_benchmark_result(const BenchmarkResult& res) {
    nlohmann::json output;
    auto [func_id, hparams] = parse_hyperparams(res.function_name);
    nlohmann::json::array_t n_vals, time_vals_mean, price_vals, hyper, time_vals_std, all_times;
    double std_time = 0., mean_time = 0.;
    for (const auto& [n_steps, time] : res.execution_times) {
        n_vals.push_back(n_steps);
        std::tie(mean_time, std_time) = mean_and_std(time);
        time_vals_mean.push_back(mean_time);
        time_vals_std.push_back(std_time);
        nlohmann::json::array_t time_at_n;
        for (auto ctime : time) {
            time_at_n.push_back(ctime);
        }
        all_times.push_back(time_at_n);
        price_vals.push_back(std::round(res.prices.at(n_steps)[0] * 1e6) / 1e6);
    }
    nlohmann::json::array_t hyper_params;
    for (const auto& hparam : hparams) {
        hyper_params.push_back(hparam);
    }
    nlohmann::json::array_t sanity_checks;
    for (auto& [pricing_input, expected, price] : res.sanity_check_results) {
        nlohmann::json sanity_check;
        sanity_check["test_name"] = pricing_input.name;
        sanity_check["expected"] = expected;
        sanity_check["price"] = price;
        sanity_checks.push_back(sanity_check);
    }

    output = nlohmann::json{{"id", res.function_name},
                            {"function_id", func_id},
                            {"do_pass_sanity_check", res.pass_sanity_check() ? "true" : "false"},
                            {"sanity_check", sanity_checks},
                            {"hyperparams", hyper_params},
                            {"all_times", all_times},
                            {"runs",
                             {{"n", n_vals},
                              {"time_ms_mean", time_vals_mean},
                              {"time_ms_std", time_vals_std},
                              {"price", price_vals}}}};

    return output;
}

nlohmann::json dump_benchmark_results_json(const std::vector<BenchmarkResult>& results) {
    nlohmann::json output;
    output["runs"] = nlohmann::json::array();

    if (!results.empty()) {
        auto run = results[0].run;
        output["benchmark_parameters"] = dump_run_json(run);
    }

    for (const auto& res : results) {
        output["runs"].push_back(dump_benchmark_result(res));
    }
    return output;
}
