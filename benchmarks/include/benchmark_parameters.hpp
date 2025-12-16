#pragma once

#include <iostream>
#include <map>
#include <string>
#include <functional>

#include "constants.hpp"


inline auto Constant_Additive_NStep = [](int step) {
    return [step](int ) {
        return step;
    };
};

inline auto Constant_Multiplicative_NStep = [](int step) {
    return [step](int current_n) {
        return current_n*step-current_n;
    };
};

inline auto NStep_125 = []() {
    return [](int current_n) {
        int div = current_n, rem;
        do {
            rem = div % 10;
            div = div / 10; 
        } while (div != 0);
        if (rem == 2)
            return current_n/2*3;
        else
            return current_n;
    };
};

inline double mean(std::vector<double> v) {
    double mean = 0.;
    for (size_t i = 0; i < v.size(); i++) mean += v[i];
    mean /= v.size();
    return mean;
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
                 OptionType type, std::string name)
        : name(name), S(S), K(K), T(T), r(r), sigma(sigma), q(q), n(n), type(type) {}
};

class Run {
   public:
    double S;
    double K;
    double T;
    double r;
    double sigma;
    double q;
    int nstart;
    int nend;
    std::function<int(int)> nstep;
    int nrepetition_at_step;
    int n_check_stop_after_repetition;
    OptionType type;


    Run()
        : S(0),
          K(0),
          T(0),
          r(0),
          sigma(0),
          q(0),
          nstart(0),
          nend(0),
          nstep(Constant_Additive_NStep(0)),
          nrepetition_at_step(1),
          type(OptionType::Call) {}

    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        std::function<int(int)> nstep, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(nstep),
          nrepetition_at_step(1),
          n_check_stop_after_repetition(1),
          type(type) {}
        
    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        std::function<int(int)> nstep, int nrepetition_at_step, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(nstep),
          nrepetition_at_step(nrepetition_at_step),
          n_check_stop_after_repetition(nrepetition_at_step),
          type(type) {}



    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        std::function<int(int)> nstep, int nrepetition_at_step, int n_check_stop_after_repetition, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(nstep),
          nrepetition_at_step(nrepetition_at_step),
          n_check_stop_after_repetition(n_check_stop_after_repetition),
          type(type) {}
};

inline std::string to_string(Run run) {
    return "S=" + std::to_string(run.S) + ", K=" + std::to_string(run.K) +
           ", T=" + std::to_string(run.T) + ", r=" + std::to_string(run.r) +
           ", sigma=" + std::to_string(run.sigma) + ", q=" + std::to_string(run.q) +
           ", nstart=" + std::to_string(run.nstart) + ", nend=" + std::to_string(run.nend) +
           ", nrepetition_at_step=" + std::to_string(run.nrepetition_at_step) +
           ", n_check_stop_after_repetition=" + std::to_string(run.n_check_stop_after_repetition) +
           ", type=" + to_string(run.type);
}

extern std::map<std::string, Run> BENCHMARK_PARAMETERS;

inline void list_benchmark_parameters() {
    std::cout << "Available benchmark parameters identifiers:\n";
    for (const auto& [name, run] : BENCHMARK_PARAMETERS) {
        std::cout << "  - " << name << ": ";
        std::cout << to_string(run) << "\n";
    }
}
