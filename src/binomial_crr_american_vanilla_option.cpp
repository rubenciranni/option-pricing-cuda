#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include "cpu/binomial_crr_american_vanilla_option_cpu.hpp"
#include "constants.hpp"

std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key[0] == '-' && i + 1 < argc) {
            args[key] = argv[i + 1];
            ++i;
        }
    }
    return args;
}

bool checkRequiredArgs(const std::unordered_map<std::string,std::string>& args) {
    const std::string required[] = {"-type","-S","-K","-T","-r","-sigma","-q","-n"};
    for (const auto& key : required) {
        if (args.find(key) == args.end()) {
            std::cerr << "Error: Missing required argument " << key << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    // Check required arguments
    if (!checkRequiredArgs(args)) {
        std::cerr << "Usage: ./binomial_crr_american_vanilla_option -type <call|put> -S <stock> -K <strike> "
                     "-T <maturity> -r <rate> -sigma <volatility> -q <dividend> -n <steps>\n";
        return 1;
    }

    // Parse arguments
    std::string t = args["-type"];
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    OptionType type;
    if (t == "call") type = OptionType::Call;
    else if (t == "put") type = OptionType::Put;
    else {
        std::cerr << "Error: Invalid option type. Use 'call' or 'put'.\n";
        return 1;
    }

    double S     = std::atof(args["-S"].c_str());
    double K     = std::atof(args["-K"].c_str());
    double T     = std::atof(args["-T"].c_str());
    double r     = std::atof(args["-r"].c_str());
    double sigma = std::atof(args["-sigma"].c_str());
    double q     = std::atof(args["-q"].c_str());
    int n        = std::atoi(args["-n"].c_str());

    double price = binomial_crr_american_vanilla_option_cpu(T, S, K, r, sigma, q, n, type);

    std::cout << (type == OptionType::Call ? "American Call Price: " : "American Put Price: ")
              << price << std::endl;

    return 0;
}
