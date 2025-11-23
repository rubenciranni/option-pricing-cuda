#include "benchmark.hpp"

#include <chrono>

#include "backends/cpu/vanilla_american_binomial_cpu.hpp"
#include "backends/cuda/vanilla_american_binomial_cuda.cuh"
#include "backends/hyperparams.hpp"
#include "benchmark_parameters.hpp"
#include "function_registry.hpp"
#include "run_random_generator.hpp"
#include "sanity_checker.hpp"

// clang-format off
std::map<std::string, Run> BENCHMARK_PARAMETERS = {
    {"xs-cpu", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 8, 8, 8, 1, OptionType::Put)},
    {"xs-cuda", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 10001, 10000, 1, OptionType::Put)},
    {"s-single", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 1000, 2000, 1000, 1, OptionType::Put)},
    {"s-repeat", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 1000, 1001, 1000, 5, OptionType::Put)},
    {"m-single", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 20000, 10000, 1, OptionType::Put)},
    {"m-repeat", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 20000, 10000, 1000, OptionType::Put)},
    {"l-repeat", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 210000, 100000, 5, OptionType::Put)},
    {"l-single", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 210000, 100000, 1, OptionType::Put)},
    {"xl-repeat", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 10000, 260000, 50000, 10, OptionType::Put)},
    {"xxl-single", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 250000, 250000, 30000, 1, OptionType::Put)},
};

std::map<std::string, PricingFunction> FUNCTION_REGISTRY = {
    {"vanilla_american_binomial_cpu_naive", vanilla_american_binomial_cpu_naive},
    {"vanilla_american_binomial_cpu_trimotm", vanilla_american_binomial_cpu_trimotm},
    {"vanilla_american_binomial_cpu_trimotm_stprcmp", vanilla_american_binomial_cpu_trimotm_stprcmp},
    {"vanilla_american_binomial_cpu_trimotm_trimeeoff_stprcmp", vanilla_american_binomial_cpu_trimotm_trimeeoff_stprcmp},
    {"vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp", vanilla_american_binomial_cpu_trimotm_trimeeon_stprcmp},
    {"vanilla_american_binomial_openmp_naive", vanilla_american_binomial_openmp_naive},
    {"vanilla_american_binomial_cuda_naive", vanilla_american_binomial_cuda_naive},
    {"vanilla_american_binomial_cuda_stprcmp", vanilla_american_binomial_cuda_stprcmp},
    {"vanilla_american_binomial_cuda_bkdstprcmp", vanilla_american_binomial_cuda_bkdstprcmp},
    {"vanilla_american_binomial_cuda_stprcmp_yunroll_vtile", vanilla_american_binomial_cuda_stprcmp_yunroll_vtile<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_YUNROLL_VTILE>},
    {"vanilla_american_binomial_cuda_stprcmp_xunroll_vprftc", vanilla_american_binomial_cuda_stprcmp_xunroll_vprftc<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XUNROLL_VPRFTC>},
    {"vanilla_american_binomial_cuda_stprcmp_xunroll_stvtile", vanilla_american_binomial_cuda_stprcmp_xunroll_stvtile<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XUNROLL_STVTILE>},
    {"vanilla_american_binomial_cuda_stprcmp_xyunroll_vprftc", vanilla_american_binomial_cuda_stprcmp_xyunroll_vprftc<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XYUNROLL_VPRFTC>},
    {"vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc", vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc},
    {"vanilla_american_binomial_cuda_stprcmp_xyunroll_stvprftc", vanilla_american_binomial_cuda_stprcmp_xyunroll_stvprftc<DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XYUNROLL_STVPRFTC>},
    {"vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc_trimotm", vanilla_american_binomial_cuda_stprcmp_xyunroll_stvtile_vprftc_trimotm},
    {"vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile", vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile},
    {"vanilla_american_binomial_cuda_bkdstprcmp_xovlpunroll_vtile", vanilla_american_binomial_cuda_bkdstprcmp_xovlpunroll_vtile<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000>},
    {"vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm", vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000>},
    {"vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm", vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>},
    {"vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_malloc", vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_malloc},


    #ifdef DO_CARTESIAN_PRODUCT
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_stprcmp_yunroll_vtile)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_STVTILE
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_stprcmp_xunroll_stvtile)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_VPRFTC
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_stprcmp_xunroll_vprftc)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_STVPRFTC
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_stprcmp_xyunroll_stvprftc)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_VPRFTC
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_stprcmp_xyunroll)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_bkdstprcmp_xovlpunroll_vtile)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm)
        #endif
        #ifdef DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_TRIMOTM_TRIMEEOFF
            APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_bkdstprcmp_xovlpunroll_vtile_trimotm_trimeeoff)
        #endif
    #endif
};
// clang-format on

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
            SanityCheckResults sanity_check;
            if (!skip_sanity_checks) {
                sanity_check = sanity_checker.run_single_all_sanity_checks(func);
            }

            BenchmarkResult result(data, benchmark_parameters, {}, name, reference_function_name,
                                   sanity_check);
            for (int n = data.nstart; n <= data.nend; n += data.nstep) {
                for (int _ = 0; _ < data.nrepetition_at_step; _++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    double price =
                        func(data.S, data.K, data.T, data.r, data.sigma, data.q, n, data.type);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration = end - start;

                    result.execution_times[n].push_back(duration.count());
                    result.prices[n].push_back(price);
                }
            }
            results.push_back(result);
        }
    }
    return results;
}

std::vector<std::vector<BenchmarkResult>> random_benchmark(
    const std::string& filter_function_name, const std::string& reference_function_name,
    const int n_random_runs, bool skip_sanity_checks) {
    if (FUNCTION_REGISTRY.find(reference_function_name) == FUNCTION_REGISTRY.end()) {
        std::cerr << "Reference function '" << reference_function_name
                  << "' not found in function registry.\n";
        return {};
    }

    SanityChecker sanity_checker(reference_function_name,
                                 FUNCTION_REGISTRY[reference_function_name]);

    std::map<std::string, SanityCheckResults> sanity_checks_map;
    std::vector<std::vector<BenchmarkResult>> results;
    for (const auto& [name, func] : FUNCTION_REGISTRY) {
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            // Measure execution time
            SanityCheckResults sanity_check;
            if (!skip_sanity_checks) {
                sanity_check = sanity_checker.run_single_all_sanity_checks(func);
                sanity_checks_map[name] = sanity_check;
            }
        }
    }
    for (const auto& run :
         RunGenerator().generateRandomRuns(n_random_runs, 1000, 2000, 100, 5, OptionType::Put)) {
        std::vector<BenchmarkResult> results_per_run;
        for (const auto& [name, func] : FUNCTION_REGISTRY) {
            // filter_function_name is a substring match
            if (name.find(filter_function_name) != std::string::npos ||
                filter_function_name.empty()) {
                BenchmarkResult result(run, "random_sampled", {}, name, reference_function_name,
                                       sanity_checks_map[name]);
                for (int n = run.nstart; n <= run.nend; n += run.nstep) {
                    for (int _ = 0; _ < run.nrepetition_at_step; _++) {
                        auto start = std::chrono::high_resolution_clock::now();
                        double price =
                            func(run.S, run.K, run.T, run.r, run.sigma, run.q, n, run.type);
                        auto end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> duration = end - start;

                        result.execution_times[n].push_back(duration.count());
                        result.prices[n].push_back(price);
                    }
                }
                results_per_run.push_back(result);
            }
            results.push_back(results_per_run);
        }
    }
    return results;
}

std::map<std::string, BatchPricingFunction> BATCH_FUNCTION_REGISTRY = {
    {"vanilla_american_binomial_cuda_batch_naive", vanilla_american_binomial_cuda_batch_naive},
    {"vanilla_american_binomial_cuda_batch_stprcmp", vanilla_american_binomial_cuda_batch_stprcmp},
    {"vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm", vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm<DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE>}
};

std::vector<BatchBenchmarkResult> batch_random_benchmark(const std::string& filter_function_name,
                                                         const std::string& reference_function_name,
                                                         const int n_random_runs, const int n,
                                                         bool skip_sanity_checks) {
    if (FUNCTION_REGISTRY.find(reference_function_name) == FUNCTION_REGISTRY.end()) {
        std::cerr << "Reference function '" << reference_function_name
                  << "' not found in function registry.\n";
        return {};
    }

    SanityChecker sanity_checker(reference_function_name,
                                 FUNCTION_REGISTRY[reference_function_name]);

    std::map<std::string, SanityCheckResults> sanity_checks_map;
    std::vector<BatchBenchmarkResult> results;
    for (const auto& [name, func] : BATCH_FUNCTION_REGISTRY) {
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            // Measure execution time
            SanityCheckResults sanity_check;
            if (!skip_sanity_checks) {
                sanity_check = sanity_checker.run_single_all_sanity_checks_batch(func);
                sanity_checks_map[name] = sanity_check;
            }
        }
    }
    std::vector<PricingInput> runs =
        RunGenerator().generateRandomPricingInput(n_random_runs, n, OptionType::Put);
    for (const auto& [name, func] : BATCH_FUNCTION_REGISTRY) {
        // filter_function_name is a substring match
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            BatchBenchmarkResult result(runs, {}, sanity_checks_map[name], name,
                                        reference_function_name);
            for (int _ = 0; _ < n_random_runs; _++) {
                std::vector<double> out(n_random_runs);
                auto start = std::chrono::high_resolution_clock::now();
                func(runs, out);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;

                result.execution_times.push_back(duration.count());
            }
            results.push_back(result);
        }
    }
    return results;
}
