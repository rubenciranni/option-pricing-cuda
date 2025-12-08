
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "benchmark_parameters.hpp"

enum class SamplingStrategy { UNIFORM, STRATIFIED, SOBOL };

class RunGenerator {
   private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_01;
    std::uniform_int_distribution<int> strategy_dist;

    // Sobol sequence generation (simplified van der Corput)
    std::vector<double> generateSobolSequence(int n, int dimension) {
        std::vector<double> sequence(n);

        for (int i = 0; i < n; ++i) {
            double value = 0.0;
            double base = 0.5;
            int num = i + 1;

            // Dimension-specific scrambling
            num = num ^ (dimension * 0x9E3779B9);

            while (num > 0) {
                if (num & 1) {
                    value += base;
                }
                base *= 0.5;
                num >>= 1;
            }
            sequence[i] = value;
        }
        return sequence;
    }

    Run sampleUniform() {
        Run run;

        run.S = 50.0 + uniform_01(rng) * 100.0;     // [50, 150]
        run.K = 50.0 + uniform_01(rng) * 100.0;     // [50, 150]
        run.T = 0.08 + uniform_01(rng) * 1.92;      // [0.08, 2.0]
        run.r = 0.01 + uniform_01(rng) * 0.04;      // [0.01, 0.05]
        run.sigma = 0.15 + uniform_01(rng) * 0.45;  // [0.15, 0.60]
        run.q = uniform_01(rng) * 0.04;             // [0.0, 0.04]

        return run;
    }

    Run sampleStratified() {
        Run run;

        // Randomly choose a regime
        std::uniform_int_distribution<int> regime_dist(0, 2);
        int regime = regime_dist(rng);

        if (regime == 0) {
            // Low volatility regime
            run.S = 100.0;
            run.K = 90.0 + uniform_01(rng) * 20.0;      // [90, 110]
            run.T = 0.25 + uniform_01(rng) * 0.75;      // [0.25, 1.0]
            run.r = 0.02 + uniform_01(rng) * 0.02;      // [0.02, 0.04]
            run.sigma = 0.15 + uniform_01(rng) * 0.10;  // [0.15, 0.25]
            run.q = uniform_01(rng) * 0.03;             // [0.0, 0.03]
        } else if (regime == 1) {
            // High volatility regime
            run.S = 100.0;
            run.K = 85.0 + uniform_01(rng) * 30.0;      // [85, 115]
            run.T = 0.25 + uniform_01(rng) * 0.75;      // [0.25, 1.0]
            run.r = 0.01 + uniform_01(rng) * 0.04;      // [0.01, 0.05]
            run.sigma = 0.40 + uniform_01(rng) * 0.20;  // [0.40, 0.60]
            run.q = uniform_01(rng) * 0.02;             // [0.0, 0.02]
        } else {
            // Deep ITM/OTM (edge cases)
            run.S = 100.0;

            bool is_deep_itm = (uniform_01(rng) < 0.5);
            if (is_deep_itm) {
                run.K = 70.0 + uniform_01(rng) * 15.0;  // [70, 85] - deep ITM
            } else {
                run.K = 115.0 + uniform_01(rng) * 15.0;  // [115, 130] - deep OTM
            }

            run.T = 0.08 + uniform_01(rng) * 1.92;      // [0.08, 2.0]
            run.r = 0.02 + uniform_01(rng) * 0.02;      // [0.02, 0.04]
            run.sigma = 0.20 + uniform_01(rng) * 0.20;  // [0.20, 0.40]
            run.q = uniform_01(rng) * 0.03;             // [0.0, 0.03]
        }

        return run;
    }

    Run sampleSobol(int sample_index) {
        Run run;

        // Generate one sample from Sobol sequence
        std::vector<double> sobol_values(6);
        for (int dim = 0; dim < 6; ++dim) {
            auto seq = generateSobolSequence(sample_index + 1, dim);
            sobol_values[dim] = seq[sample_index];
        }

        run.S = 50.0 + sobol_values[0] * 100.0;     // [50, 150]
        run.K = 50.0 + sobol_values[1] * 100.0;     // [50, 150]
        run.T = 0.08 + sobol_values[2] * 1.92;      // [0.08, 2.0]
        run.r = 0.01 + sobol_values[3] * 0.04;      // [0.01, 0.05]
        run.sigma = 0.15 + sobol_values[4] * 0.45;  // [0.15, 0.60]
        run.q = sobol_values[5] * 0.04;             // [0.0, 0.04]

        return run;
    }

   public:
    RunGenerator(unsigned int seed = 42) : rng(seed), uniform_01(0.0, 1.0), strategy_dist(0, 2) {}

    // Generate a random run by randomly picking a sampling strategy
    Run generateRandomRun(int nstart, int nend, int nstep, int nrepetition_at_step,
                          OptionType type) {
        // Randomly select sampling strategy
        SamplingStrategy strategy = static_cast<SamplingStrategy>(strategy_dist(rng));

        Run run;

        // Generate parameters based on selected strategy
        if (strategy == SamplingStrategy::UNIFORM) {
            run = sampleUniform();
        } else if (strategy == SamplingStrategy::STRATIFIED) {
            run = sampleStratified();
        } else {  // SOBOL
            static int sobol_counter = 0;
            run = sampleSobol(sobol_counter++);
        }

        // Set the fixed parameters
        run.nstart = nstart;
        run.nend = nend;
        run.nstep = Constant_Additive_NStep(nstep);
        run.nrepetition_at_step = nrepetition_at_step;
        run.type = type;

        return run;
    }

    std::vector<PricingInput> generateRandomPricingInput(int n_runs, int n, OptionType type) {
        std::vector<PricingInput> runs;
        runs.reserve(n_runs);

        for (int i = 0; i < n_runs; ++i) {
            auto run =generateRandomRun(n, n, n, 1, type);
            runs.push_back(PricingInput(run.S, run.K, run.T, run.r, run.sigma, run.q, n, type));
        }

        return runs;
    }

    std::vector<Run> generateRandomRuns(int n_runs, int nstart, int nend, int nstep,
                                        int nrepetition_at_step, OptionType type) {
        std::vector<Run> runs;
        runs.reserve(n_runs);

        for (int i = 0; i < n_runs; ++i) {
            runs.push_back(generateRandomRun(nstart, nend, nstep, nrepetition_at_step, type));
        }

        return runs;
    }
};
