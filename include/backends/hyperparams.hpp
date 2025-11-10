#pragma once

struct Hyperparams {
    const int THREADS_PER_BLOCK;
    const int UNROLL_FACTOR;
    const int OUTPUTS_PER_THREAD;
    const int MAX_LEVEL_SIZE;

    constexpr Hyperparams(const int THREADS_PER_BLOCK, const int UNROLL_FACTOR, const int OUTPUTS_PER_THREAD): 
        THREADS_PER_BLOCK(THREADS_PER_BLOCK), 
        UNROLL_FACTOR(UNROLL_FACTOR) ,
        OUTPUTS_PER_THREAD(OUTPUTS_PER_THREAD),
        MAX_LEVEL_SIZE(UNROLL_FACTOR + OUTPUTS_PER_THREAD - 1)
        {};
};


inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_TILE(1024, -1, 2);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_UNROLL_TILE(128, 7, -1);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_UNROLL(256, 7, -1);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_XY_UNROLL(256, 16, 2);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_XY_UNROLL_NEW(512, 2, 2);