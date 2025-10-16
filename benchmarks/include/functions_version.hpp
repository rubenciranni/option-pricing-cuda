#pragma once
#include <map>
#include <functional>
#include "constants.hpp"
#include <string>


typedef std::function<double(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type)>  PricingFunction;


extern std::map<std::string, PricingFunction> FUNCTIONS;