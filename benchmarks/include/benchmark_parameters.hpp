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
    int nstep;
    std::function<int(int)> nstep_fct;
    int nrepetition_at_step;
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
          nstep(0),
          nrepetition_at_step(1),
          type(OptionType::Call) {}

    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        int nstep, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(nstep),
          nstep_fct(Constant_Additive_NStep(nstep)),
          nrepetition_at_step(1),
          type(type) {}
        
    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        int nstep, int nrepetition_at_step, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(nstep),
          nstep_fct(Constant_Additive_NStep(nstep)),
          nrepetition_at_step(nrepetition_at_step),
          type(type) {}

    Run(double S, double K, double T, double r, double sigma, double q, int nstart, int nend,
        std::function<int(int)> nstep_fct, int nrepetition_at_step, OptionType type)
        : S(S),
          K(K),
          T(T),
          r(r),
          sigma(sigma),
          q(q),
          nstart(nstart),
          nend(nend),
          nstep(-1),
          nstep_fct(nstep_fct),
          nrepetition_at_step(nrepetition_at_step),
          type(type) {}

};

inline std::string to_string(Run run) {
    return "S=" + std::to_string(run.S) + ", K=" + std::to_string(run.K) +
           ", T=" + std::to_string(run.T) + ", r=" + std::to_string(run.r) +
           ", sigma=" + std::to_string(run.sigma) + ", q=" + std::to_string(run.q) +
           ", nstart=" + std::to_string(run.nstart) + ", nend=" + std::to_string(run.nend) +
           ", nstep=" + std::to_string(run.nstep) + ", nrepetition_at_step=" + std::to_string(run.nrepetition_at_step) +
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
