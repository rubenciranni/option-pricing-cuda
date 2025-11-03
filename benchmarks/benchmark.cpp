#include "benchmark.hpp"

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"

std::vector<Result> benchmark(const std::string& filter_function_name,
                              const std::string& benchmark_parameters, bool no_verify) {
    if (BENCHMARK_PARAMETERS.find(benchmark_parameters) == BENCHMARK_PARAMETERS.end()) {
        std::cerr << "Benchmark parameters identifier not found: " << benchmark_parameters << "\n";
        return {};
    }
    // TestFunctionSanityChecks sanity_checker("vanilla_american_binomial_cpu_naive",
    //                                         vanilla_american_binomial_cpu_naive);

    TestFunctionSanityChecks sanity_checker("vanilla_american_binomial_cuda_unroll",
                                            vanilla_american_binomial_cuda_unroll);
    const Run& data = BENCHMARK_PARAMETERS[benchmark_parameters];
    std::vector<Result> results;
    for (const auto& [name, func] : FUNCTIONS) {
        // filter_function_name is a substring match
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            // Measure execution time
            bool sanity_check =
                no_verify || sanity_checker.run_single_all_sanity_checks(name, func);
            Result result(data, benchmark_parameters, {}, name, sanity_check);
            for (int n = data.nstart; n <= data.nend; n += data.nstep) {
                auto start = std::chrono::high_resolution_clock::now();
                func(data.S, data.K, data.T, data.r, data.sigma, data.q, n, data.type);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                result.execution_times[n] = duration.count();
            }
            results.push_back(result);
        }
    }
    return results;
}
