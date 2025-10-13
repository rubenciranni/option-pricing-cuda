#pragma once

#include <string>

enum class OptionType {
    Call,
    Put
};

inline const OptionType DEFAULT_OPTION_TYPE = OptionType::Call;

constexpr int option_type_sign(OptionType type) noexcept {
    return (type == OptionType::Call) ? 1 : -1;
}

inline std::string to_string(OptionType type) {
    return (type == OptionType::Call) ? "Call" : "Put";
}
