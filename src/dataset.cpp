#include "dataset.hpp"

std::map<std::string, Run> DATASET= {
    {"easy", Run(100, 100, 0.5, 0.03, 0.2, 0.015, 1000, 2000, 1000, OptionType::Put)},
};


void list_datasets() {
    std::cout << "Available datasets:\n";
    for (const auto& [name, run] : DATASET) {
        std::cout << "  - " << name << ": ";
        std::cout << to_string(run) << "\n";
    }
}






