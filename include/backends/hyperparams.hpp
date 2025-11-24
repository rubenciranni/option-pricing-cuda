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
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE_10000(128, 37, -1);
inline constexpr Hyperparams DEFAULT_HYPERPARAMS_CUDA_BKDSTPRCMP_XOVLPUNROLL_SHUFFLE(128, 50, -1);

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
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM

#ifdef DO_CARTESIAN_PRODUCT
#define HYPERPARAMS_CART_PRODUCT(X, Y) CART_PROD_1(X, Y)

#define CART_PROD_1(X, Y)      \
    CART_PROD_2(D, 128, X, Y)

#define CART_PROD_2(ID, A, X, Y) \
    CART_PROD_3(ID##0,   A, 1,   X, Y) \
    CART_PROD_3(ID##1,   A, 2,   X, Y) \
    CART_PROD_3(ID##2,   A, 3,   X, Y) \
    CART_PROD_3(ID##3,   A, 4,   X, Y) \
    CART_PROD_3(ID##4,   A, 5,   X, Y) \
    CART_PROD_3(ID##5,   A, 6,   X, Y) \
    CART_PROD_3(ID##6,   A, 7,   X, Y) \
    CART_PROD_3(ID##7,   A, 8,   X, Y) \
    CART_PROD_3(ID##8,   A, 9,   X, Y) \
    CART_PROD_3(ID##9,   A, 10,  X, Y) \
    CART_PROD_3(ID##10,  A, 11,  X, Y) \
    CART_PROD_3(ID##11,  A, 12,  X, Y) \
    CART_PROD_3(ID##12,  A, 13,  X, Y) \
    CART_PROD_3(ID##13,  A, 14,  X, Y) \
    CART_PROD_3(ID##14,  A, 15,  X, Y) \
    CART_PROD_3(ID##15,  A, 16,  X, Y) \
    CART_PROD_3(ID##16,  A, 17,  X, Y) \
    CART_PROD_3(ID##17,  A, 18,  X, Y) \
    CART_PROD_3(ID##18,  A, 19,  X, Y) \
    CART_PROD_3(ID##19,  A, 20,  X, Y) \
    CART_PROD_3(ID##20,  A, 21,  X, Y) \
    CART_PROD_3(ID##21,  A, 22,  X, Y) \
    CART_PROD_3(ID##22,  A, 23,  X, Y) \
    CART_PROD_3(ID##23,  A, 24,  X, Y) \
    CART_PROD_3(ID##24,  A, 25,  X, Y) \
    CART_PROD_3(ID##25,  A, 26,  X, Y) \
    CART_PROD_3(ID##26,  A, 27,  X, Y) \
    CART_PROD_3(ID##27,  A, 28,  X, Y) \
    CART_PROD_3(ID##28,  A, 29,  X, Y) \
    CART_PROD_3(ID##29,  A, 30,  X, Y) \
    CART_PROD_3(ID##30,  A, 31,  X, Y) \
    CART_PROD_3(ID##31,  A, 32,  X, Y) \
    CART_PROD_3(ID##32,  A, 33,  X, Y) \
    CART_PROD_3(ID##33,  A, 34,  X, Y) \
    CART_PROD_3(ID##34,  A, 35,  X, Y) \
    CART_PROD_3(ID##35,  A, 36,  X, Y) \
    CART_PROD_3(ID##36,  A, 37,  X, Y) \
    CART_PROD_3(ID##37,  A, 38,  X, Y) \
    CART_PROD_3(ID##38,  A, 39,  X, Y) \
    CART_PROD_3(ID##39,  A, 40,  X, Y) \
    CART_PROD_3(ID##40,  A, 41,  X, Y) \
    CART_PROD_3(ID##41,  A, 42,  X, Y) \
    CART_PROD_3(ID##42,  A, 43,  X, Y) \
    CART_PROD_3(ID##43,  A, 44,  X, Y) \
    CART_PROD_3(ID##44,  A, 45,  X, Y) \
    CART_PROD_3(ID##45,  A, 46,  X, Y) \
    CART_PROD_3(ID##46,  A, 47,  X, Y) \
    CART_PROD_3(ID##47,  A, 48,  X, Y) \
    CART_PROD_3(ID##48,  A, 49,  X, Y) \
    CART_PROD_3(ID##49,  A, 50,  X, Y) \
    CART_PROD_3(ID##50,  A, 51,  X, Y) \
    CART_PROD_3(ID##51,  A, 52,  X, Y) \
    CART_PROD_3(ID##52,  A, 53,  X, Y) \
    CART_PROD_3(ID##53,  A, 54,  X, Y) \
    CART_PROD_3(ID##54,  A, 55,  X, Y) \
    CART_PROD_3(ID##55,  A, 56,  X, Y) \
    CART_PROD_3(ID##56,  A, 57,  X, Y) \
    CART_PROD_3(ID##57,  A, 58,  X, Y) \
    CART_PROD_3(ID##58,  A, 59,  X, Y) \
    CART_PROD_3(ID##59,  A, 60,  X, Y) \
    CART_PROD_3(ID##60,  A, 61,  X, Y) \
    CART_PROD_3(ID##61,  A, 62,  X, Y) \
    CART_PROD_3(ID##62,  A, 63,  X, Y) \
    CART_PROD_3(ID##63,  A, 64,  X, Y) \
    CART_PROD_3(ID##64,  A, 65,  X, Y) \
    CART_PROD_3(ID##65,  A, 66,  X, Y) \
    CART_PROD_3(ID##66,  A, 67,  X, Y) \
    CART_PROD_3(ID##67,  A, 68,  X, Y) \
    CART_PROD_3(ID##68,  A, 69,  X, Y) \
    CART_PROD_3(ID##69,  A, 70,  X, Y) \
    CART_PROD_3(ID##70,  A, 71,  X, Y) \
    CART_PROD_3(ID##71,  A, 72,  X, Y) \
    CART_PROD_3(ID##72,  A, 73,  X, Y) \
    CART_PROD_3(ID##73,  A, 74,  X, Y) \
    CART_PROD_3(ID##74,  A, 75,  X, Y) \
    CART_PROD_3(ID##75,  A, 76,  X, Y) \
    CART_PROD_3(ID##76,  A, 77,  X, Y) \
    CART_PROD_3(ID##77,  A, 78,  X, Y) \
    CART_PROD_3(ID##78,  A, 79,  X, Y) \
    CART_PROD_3(ID##79,  A, 80,  X, Y) \
    CART_PROD_3(ID##80,  A, 81,  X, Y) \
    CART_PROD_3(ID##81,  A, 82,  X, Y) \
    CART_PROD_3(ID##82,  A, 83,  X, Y) \
    CART_PROD_3(ID##83,  A, 84,  X, Y) \
    CART_PROD_3(ID##84,  A, 85,  X, Y) \
    CART_PROD_3(ID##85,  A, 86,  X, Y) \
    CART_PROD_3(ID##86,  A, 87,  X, Y) \
    CART_PROD_3(ID##87,  A, 88,  X, Y) \
    CART_PROD_3(ID##88,  A, 89,  X, Y) \
    CART_PROD_3(ID##89,  A, 90,  X, Y) \
    CART_PROD_3(ID##90,  A, 91,  X, Y) \
    CART_PROD_3(ID##91,  A, 92,  X, Y) \
    CART_PROD_3(ID##92,  A, 93,  X, Y) \
    CART_PROD_3(ID##93,  A, 94,  X, Y) \
    CART_PROD_3(ID##94,  A, 95,  X, Y) \
    CART_PROD_3(ID##95,  A, 96,  X, Y) \
    CART_PROD_3(ID##96,  A, 97,  X, Y) \
    CART_PROD_3(ID##97,  A, 98,  X, Y) \
    CART_PROD_3(ID##98,  A, 99,  X, Y) \
    CART_PROD_3(ID##99,  A, 100, X, Y) \
    CART_PROD_3(ID##100, A, 101, X, Y) \
    CART_PROD_3(ID##101, A, 102, X, Y) \
    CART_PROD_3(ID##102, A, 103, X, Y) \
    CART_PROD_3(ID##103, A, 104, X, Y) \
    CART_PROD_3(ID##104, A, 105, X, Y) \
    CART_PROD_3(ID##105, A, 106, X, Y) \
    CART_PROD_3(ID##106, A, 107, X, Y) \
    CART_PROD_3(ID##107, A, 108, X, Y) \
    CART_PROD_3(ID##108, A, 109, X, Y) \
    CART_PROD_3(ID##109, A, 110, X, Y) \
    CART_PROD_3(ID##110, A, 111, X, Y) \
    CART_PROD_3(ID##111, A, 112, X, Y) \
    CART_PROD_3(ID##112, A, 113, X, Y) \
    CART_PROD_3(ID##113, A, 114, X, Y) \
    CART_PROD_3(ID##114, A, 115, X, Y) \
    CART_PROD_3(ID##115, A, 116, X, Y) \
    CART_PROD_3(ID##116, A, 117, X, Y) \
    CART_PROD_3(ID##117, A, 118, X, Y) \
    CART_PROD_3(ID##118, A, 119, X, Y) \
    CART_PROD_3(ID##119, A, 120, X, Y) \
    CART_PROD_3(ID##120, A, 121, X, Y) \
    CART_PROD_3(ID##121, A, 122, X, Y) \
    CART_PROD_3(ID##122, A, 123, X, Y) \
    CART_PROD_3(ID##123, A, 124, X, Y) \
    CART_PROD_3(ID##124, A, 125, X, Y) \
    CART_PROD_3(ID##125, A, 126, X, Y) \
    CART_PROD_3(ID##126, A, 127, X, Y) \

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
