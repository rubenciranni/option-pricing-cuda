#include "benchmark.hpp"

#include <chrono>

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "benchmark_parameters.hpp"
#include "function_registry.hpp"
#include "sanity_checker.hpp"




std::map<std::string, Run> BENCHMARK_PARAMETERS = {
    {"debug", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 20, 20, 20, OptionType::Put)},
    {"easy", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 1000, 2000, 1000, OptionType::Put)},
    {"cuda_debug", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 10001, 10000, OptionType::Put)},
    {"hard", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 20000, 10000, OptionType::Put)},
    {"super_hard", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 250000, 250000, 30000, OptionType::Put)},
};

std::map<std::string, PricingFunction> FUNCTION_REGISTRY = {
    {"vanilla_american_binomial_cpu_naive", vanilla_american_binomial_cpu_naive},
    {"vanilla_american_binomial_cpu_remove_zeros_cache", vanilla_american_binomial_cpu_remove_zeros_cache},
    {"vanilla_american_binomial_openmp_naive", vanilla_american_binomial_openmp_naive},
    {"vanilla_american_binomial_cpu_remove_zeros", vanilla_american_binomial_cpu_remove_zeros},
    {"vanilla_american_binomial_cuda_naive", vanilla_american_binomial_cuda_naive},
    {"vanilla_american_binomial_cuda_no_sync", vanilla_american_binomial_cuda_no_sync},
    {"vanilla_american_binomial_cuda_fill", vanilla_american_binomial_cuda_fill},
    {"vanilla_american_binomial_cuda_precomputed_stock_price", vanilla_american_binomial_cuda_precomputed_stock_price},
    {"vanilla_american_binomial_cuda_tile", vanilla_american_binomial_cuda_tile<DEFAULT_HYPERPARAMS_CUDA_TILE>},
    {"vanilla_american_binomial_cuda_unroll", vanilla_american_binomial_cuda_unroll<DEFAULT_HYPERPARAMS_CUDA_UNROLL>},
    {"vanilla_american_binomial_cuda_x_y_unroll_H1", vanilla_american_binomial_cuda_x_y_unroll<DEFAULT_HYPERPARAMS_CUDA_XY_UNROLL>},
    {"vanilla_american_binomial_cuda_unroll_tile", vanilla_american_binomial_cuda_unroll_tile<DEFAULT_HYPERPARAMS_CUDA_UNROLL_TILE>},
    {"vanilla_american_binomial_cuda_x_y_unroll_new_H1", vanilla_american_binomial_cuda_x_y_unroll_new<DEFAULT_HYPERPARAMS_CUDA_XY_UNROLL_NEW>}
};

std::vector<BenchmarkResult> benchmark(const std::string& filter_function_name,
                                       const std::string& benchmark_parameters,
                                       const std::string& reference_function_name,
                                       bool skip_sanity_checks) {
    if (BENCHMARK_PARAMETERS.find(benchmark_parameters) == BENCHMARK_PARAMETERS.end()) {
        std::cerr << "Benchmark parameters identifier '" << benchmark_parameters
                  << "' not found.\n";
        return {};
    }

    if (FUNCTION_REGISTRY.find(reference_function_name) == FUNCTION_REGISTRY.end()) {
        std::cerr << "Reference function '" << reference_function_name
                  << "' not found in function registry.\n";
        return {};
    }

    SanityChecker sanity_checker(reference_function_name,
                                 FUNCTION_REGISTRY[reference_function_name]);

    const Run& data = BENCHMARK_PARAMETERS[benchmark_parameters];
    std::vector<BenchmarkResult> results;
    for (const auto& [name, func] : FUNCTION_REGISTRY) {
        // filter_function_name is a substring match
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            // Measure execution time
            bool sanity_check =
                skip_sanity_checks || sanity_checker.run_single_all_sanity_checks(name, func);
            BenchmarkResult result(data, benchmark_parameters, {}, name, sanity_check);
            for (int n = data.nstart; n <= data.nend; n += data.nstep) {
                auto start = std::chrono::high_resolution_clock::now();
                double price =
                    func(data.S, data.K, data.T, data.r, data.sigma, data.q, n, data.type);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                result.execution_times[n] = duration.count();
                result.prices[n] = price;
            }
            results.push_back(result);
        }
    }
    return results;
}
