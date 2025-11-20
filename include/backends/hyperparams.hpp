#pragma once

// clang-format off
// ==========================================
// General Hyperparameters
// ==========================================

struct Hyperparams {
    const int THREADS_PER_BLOCK;
    const int UNROLL_FACTOR;
    const int OUTPUTS_PER_THREAD;
    const int MAX_LEVEL_SIZE;

    constexpr Hyperparams(const int THREADS_PER_BLOCK, const int UNROLL_FACTOR,
                          const int OUTPUTS_PER_THREAD)
        : THREADS_PER_BLOCK(THREADS_PER_BLOCK),
          UNROLL_FACTOR(UNROLL_FACTOR),
          OUTPUTS_PER_THREAD(OUTPUTS_PER_THREAD),
          MAX_LEVEL_SIZE(UNROLL_FACTOR + OUTPUTS_PER_THREAD - 1){};
};

inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_STPRCMP_YUNROLL_VTILE(1024, -1, 2);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XUNROLL_STVTILE(128, 7, -1);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XUNROLL_VPRFTC(256, 7, -1);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XYUNROLL_VPRFTC(256, 16, 2);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_STPRCMP_XYUNROLL_STVPRFTC(512, 2, 2);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000(128, 37,
                                                                                         -1);

// ==========================================
// Grid Search
// ==========================================

// #define DO_CARTESIAN_PRODUCT
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_STVTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_VPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_STVPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_VPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_TRIMOTM

#ifdef DO_CARTESIAN_PRODUCT
#define HYPERPARAMS_CART_PRODUCT(X, Y) CART_PROD_1(X, Y)

#define CART_PROD_1(X, Y)      \
    CART_PROD_2(A, 1024, X, Y) \
    CART_PROD_2(B, 512, X, Y)  \
    CART_PROD_2(C, 256, X, Y)  \
    CART_PROD_2(D, 128, X, Y)

#define CART_PROD_2(ID, A, X, Y)    \
    CART_PROD_3(ID##0, A, 8, X, Y)  \
    CART_PROD_3(ID##1, A, 16, X, Y) \
    CART_PROD_3(ID##2, A, 32, X, Y)

#define CART_PROD_3(ID, A, B, X, Y) CART_PROD_4(ID /*##0*/, A, B, 0, X, Y)

#define CART_PROD_4(ID, A, B, C, X, Y) CART_PROD_5(ID /*##0*/, A, B, C, 0, X, Y)

#define CART_PROD_5(ID, A, B, C, D, X, Y) X(ID /*##0*/, A, B, C, D, 0, Y)

#define STR(x) #x
#define APPLY_FUNCTION(FUNC, PROD, FUNC_PARAM) PROD(FUNC, FUNC_PARAM)

#define PRODUCE_HYPERPARAMS_INSTANCES_3(ID, A, B, C, D, E, Y) \
    inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_##ID(A, B, C);
#define PRODUCE_HYPERPARAMS_INSTANCES_4(ID, A, B, C, D, E, Y) \
    inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_##ID(A, B, C, D);
#define PRODUCE_HYPERPARAMS_INSTANCES_5(ID, A, B, C, D, E, Y) \
    inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_##ID(A, B, C, D, E);

#define PRODUCE_FUNCTIONS_FOR_REGISTRY(ID, A, B, C, D, E, Y)                            \
    { STR(Y##_@H##ID##_##A##_##B##_##C##_##D##_##E), Y<GRID_SEARCH_HYPERPARAMS_##ID> },

APPLY_FUNCTION(PRODUCE_HYPERPARAMS_INSTANCES_3, HYPERPARAMS_CART_PRODUCT, NULL)
#endif
