#include "benchmark.hpp"


void benchmark(const std::string& filter_function_name, const std::string& dataset) {
    if (DATASET.find(dataset) == DATASET.end()) {
        std::cerr << "Dataset not found: " << dataset << "\n";
        return;
    }

    const Run& data = DATASET[dataset];
    std::vector<Result> results;
    for (const auto& [name, func] : FUNCTIONS) {
        // filter_function_name is a substring match
        if (name.find(filter_function_name) != std::string::npos || filter_function_name.empty()) {
            // Measure execution time
            bool sanity_check = run_sanity_checks_by_function(func);
            Result result(data, dataset, {}, name,sanity_check);
            for (int n = data.nstart; n <= data.nend; n += data.nstep) {
                auto start = std::chrono::high_resolution_clock::now();
                double price = func(data.S, data.K, data.T, data.r, data.sigma, data.q, n, data.type);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                result.execution_times[n] = duration.count();
            }
            results.push_back(result);
        }

    }
    for (const auto& res : results) {
        std::cout << to_string(res) << "\n";
    }
}