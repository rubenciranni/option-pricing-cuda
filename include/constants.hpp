#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

enum class OptionType { Call, Put };
enum class ExerciseType { European, American };
enum class PricingMethod { Binomial };
enum class Backend { CPU, OpenMP, CUDA };

constexpr int option_type_sign(OptionType type) noexcept {
    return (type == OptionType::Call) ? 1 : -1;
}

inline void lowercase_transform(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
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

inline OptionType option_type_from_string(std::string& type) {
    lowercase_transform(type);
    if (type == "call")
        return OptionType::Call;
    else if (type == "put")
        return OptionType::Put;
    else
        throw std::invalid_argument("Invalid OptionType: " + type);
}

inline Backend backend_from_string(std::string& backend) {
    lowercase_transform(backend);
    if (backend == "cpu")
        return Backend::CPU;
    else if (backend == "openmp")
        return Backend::OpenMP;
    else if (backend == "cuda")
        return Backend::CUDA;
    else
        throw std::invalid_argument("Invalid Backend: " + backend);
}

inline PricingMethod pricing_method_from_string(std::string& method) {
    lowercase_transform(method);
    if (method == "binomial")
        return PricingMethod::Binomial;
    else
        throw std::invalid_argument("Invalid PricingMethod: " + method);
}

inline ExerciseType exercise_type_from_string(std::string& style) {
    lowercase_transform(style);
    if (style == "european")
        return ExerciseType::European;
    else if (style == "american")
        return ExerciseType::American;
    else
        throw std::invalid_argument("Invalid ExerciseType: " + style);
}
