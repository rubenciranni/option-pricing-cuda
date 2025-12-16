#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

enum class OptionType { Call, Put };
enum class ExerciseType { European, American };
enum class PricingMethod { Binomial };
enum class Backend { CPU, OpenMP, CUDA };
enum class OutputFormat { PPRINT, JSON };

constexpr int option_type_sign(OptionType type) noexcept {
    return (type == OptionType::Call) ? 1 : -1;
}

inline std::string lowercase_transform(const std::string& str) {
    std::string out = str;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

inline std::string to_string(OptionType type) {
    switch (type) {
        case OptionType::Call:
            return "Call";
        case OptionType::Put:
            return "Put";
        default:
            return "Unknown OptionType";
    }
}

inline std::string to_string(Backend backend) {
    switch (backend) {
        case Backend::CPU:
            return "CPU";
        case Backend::OpenMP:
            return "OpenMP";
        case Backend::CUDA:
            return "CUDA";
        default:
            return "Unknown Backend";
    }
}

inline std::string to_string(PricingMethod method) {
    switch (method) {
        case PricingMethod::Binomial:
            return "Binomial";
        default:
            return "Unknown PricingMethod";
    }
}

inline std::string to_string(ExerciseType style) {
    switch (style) {
        case ExerciseType::European:
            return "European";
        case ExerciseType::American:
            return "American";
        default:
            return "Unknown ExerciseType";
    }
}

inline OptionType option_type_from_string(const std::string& type) {
    std::string lower_type = lowercase_transform(type);
    if (lower_type == "call")
        return OptionType::Call;
    else if (lower_type == "put")
        return OptionType::Put;
    else
        throw std::invalid_argument("Invalid OptionType: " + type);
}

inline Backend backend_from_string(const std::string& backend) {
    std::string lower_backend = lowercase_transform(backend);
    if (lower_backend == "cpu")
        return Backend::CPU;
    else if (lower_backend == "openmp")
        return Backend::OpenMP;
    else if (lower_backend == "cuda")
        return Backend::CUDA;
    else
        throw std::invalid_argument("Invalid Backend: " + backend);
}

inline PricingMethod pricing_method_from_string(const std::string& method) {
    std::string lower_method = lowercase_transform(method);
    if (lower_method == "binomial")
        return PricingMethod::Binomial;
    else
        throw std::invalid_argument("Invalid PricingMethod: " + method);
}

inline OutputFormat output_format_from_string(const std::string& type) {
    std::string lower_type = lowercase_transform(type);
    if (lower_type == "pprint")
        return OutputFormat::PPRINT;
    else if (lower_type == "json")
        return OutputFormat::JSON;
    else
        throw std::invalid_argument("Invalid OutputFormat: " + type);
}

inline ExerciseType exercise_type_from_string(const std::string& style) {
    std::string lower_style = lowercase_transform(style);
    if (lower_style == "european")
        return ExerciseType::European;
    else if (lower_style == "american")
        return ExerciseType::American;
    else
        throw std::invalid_argument("Invalid ExerciseType: " + style);
}

class PricingInput {
   public:
    std::string name;
    double S;
    double K;
    double T;
    double r;
    double sigma;
    double q;
    int n;
    OptionType type;

    PricingInput() : S(0), K(0), T(0), r(0), sigma(0), q(0), n(0), type(OptionType::Call) {}

    PricingInput(double S, double K, double T, double r, double sigma, double q, int n,
                 OptionType type)
        : name(""), S(S), K(K), T(T), r(r), sigma(sigma), q(q), n(n), type(type) {}

    PricingInput(double S, double K, double T, double r, double sigma, double q, int n,
                 OptionType type, const std::string& name)
        : name(name), S(S), K(K), T(T), r(r), sigma(sigma), q(q), n(n), type(type) {}
};
