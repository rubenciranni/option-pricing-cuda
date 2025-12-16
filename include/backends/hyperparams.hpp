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

#define DO_CARTESIAN_PRODUCT
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_YUNROLL_VTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_STVTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XOVLPUNROLL_VTILE
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XUNROLL_VPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_STVPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_STPRCMP_XYUNROLL_VPRFTC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_VTILE_TRIMOTM
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_MALLOC
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_DS
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_BKDSTPRCMP_XDOVLPUNROLL_SHUFFLE_TRIMOTM_FLOAT

#ifdef DO_CARTESIAN_PRODUCT
#define HYPERPARAMS_CART_PRODUCT(X, Y) CART_PROD_1(X, Y)

#define CART_PROD_1(X, Y)           \
    CART_PROD_2_128(A, 128, X, Y)       \

#define CART_PROD_2_128(ID, A, X, Y)           \
    CART_PROD_3(ID##1, A, 1, X, Y)         \
    CART_PROD_3(ID##2, A, 2, X, Y)         \
    CART_PROD_3(ID##3, A, 3, X, Y)         \
    CART_PROD_3(ID##4, A, 4, X, Y)         \
    CART_PROD_3(ID##5, A, 5, X, Y)         \
    CART_PROD_3(ID##6, A, 6, X, Y)         \
    CART_PROD_3(ID##7, A, 7, X, Y)         \
    CART_PROD_3(ID##8, A, 8, X, Y)         \
    CART_PROD_3(ID##9, A, 9, X, Y)         \
    CART_PROD_3(ID##10, A, 10, X, Y)       \
    CART_PROD_3(ID##11, A, 11, X, Y)       \
    CART_PROD_3(ID##12, A, 12, X, Y)       \
    CART_PROD_3(ID##13, A, 13, X, Y)       \
    CART_PROD_3(ID##14, A, 14, X, Y)       \
    CART_PROD_3(ID##15, A, 15, X, Y)       \
    CART_PROD_3(ID##16, A, 16, X, Y)       \
    CART_PROD_3(ID##17, A, 17, X, Y)       \
    CART_PROD_3(ID##18, A, 18, X, Y)       \
    CART_PROD_3(ID##19, A, 19, X, Y)       \
    CART_PROD_3(ID##20, A, 20, X, Y)       \
    CART_PROD_3(ID##21, A, 21, X, Y)       \
    CART_PROD_3(ID##22, A, 22, X, Y)       \
    CART_PROD_3(ID##23, A, 23, X, Y)       \
    CART_PROD_3(ID##24, A, 24, X, Y)       \
    CART_PROD_3(ID##25, A, 25, X, Y)       \
    CART_PROD_3(ID##26, A, 26, X, Y)       \
    CART_PROD_3(ID##27, A, 27, X, Y)       \
    CART_PROD_3(ID##28, A, 28, X, Y)       \
    CART_PROD_3(ID##29, A, 29, X, Y)       \
    CART_PROD_3(ID##30, A, 30, X, Y)       \
    CART_PROD_3(ID##31, A, 31, X, Y)       \
    CART_PROD_3(ID##32, A, 32, X, Y)       \
    CART_PROD_3(ID##33, A, 33, X, Y)       \
    CART_PROD_3(ID##34, A, 34, X, Y)       \
    CART_PROD_3(ID##35, A, 35, X, Y)       \
    CART_PROD_3(ID##36, A, 36, X, Y)       \
    CART_PROD_3(ID##37, A, 37, X, Y)       \
    CART_PROD_3(ID##38, A, 38, X, Y)       \
    CART_PROD_3(ID##39, A, 39, X, Y)       \
    CART_PROD_3(ID##40, A, 40, X, Y)       \
    CART_PROD_3(ID##41, A, 41, X, Y)       \
    CART_PROD_3(ID##42, A, 42, X, Y)       \
    CART_PROD_3(ID##43, A, 43, X, Y)       \
    CART_PROD_3(ID##44, A, 44, X, Y)       \
    CART_PROD_3(ID##45, A, 45, X, Y)       \
    CART_PROD_3(ID##46, A, 46, X, Y)       \
    CART_PROD_3(ID##47, A, 47, X, Y)       \
    CART_PROD_3(ID##48, A, 48, X, Y)       \
    CART_PROD_3(ID##49, A, 49, X, Y)       \
    CART_PROD_3(ID##50, A, 50, X, Y)       \
    CART_PROD_3(ID##51, A, 51, X, Y)       \
    CART_PROD_3(ID##52, A, 52, X, Y)       \
    CART_PROD_3(ID##53, A, 53, X, Y)       \
    CART_PROD_3(ID##54, A, 54, X, Y)       \
    CART_PROD_3(ID##55, A, 55, X, Y)       \
    CART_PROD_3(ID##56, A, 56, X, Y)       \
    CART_PROD_3(ID##57, A, 57, X, Y)       \
    CART_PROD_3(ID##58, A, 58, X, Y)       \
    CART_PROD_3(ID##59, A, 59, X, Y)       \
    CART_PROD_3(ID##60, A, 60, X, Y)       \
    CART_PROD_3(ID##61, A, 61, X, Y)       \
    CART_PROD_3(ID##62, A, 62, X, Y)       \
    CART_PROD_3(ID##63, A, 63, X, Y)       \
    CART_PROD_3(ID##64, A, 64, X, Y)       \
    CART_PROD_3(ID##65, A, 65, X, Y)       \
    CART_PROD_3(ID##66, A, 66, X, Y)       \
    CART_PROD_3(ID##67, A, 67, X, Y)       \
    CART_PROD_3(ID##68, A, 68, X, Y)       \
    CART_PROD_3(ID##69, A, 69, X, Y)       \
    CART_PROD_3(ID##70, A, 70, X, Y)       \
    CART_PROD_3(ID##71, A, 71, X, Y)       \
    CART_PROD_3(ID##72, A, 72, X, Y)       \
    CART_PROD_3(ID##73, A, 73, X, Y)       \
    CART_PROD_3(ID##74, A, 74, X, Y)       \
    CART_PROD_3(ID##75, A, 75, X, Y)       \
    CART_PROD_3(ID##76, A, 76, X, Y)       \
    CART_PROD_3(ID##77, A, 77, X, Y)       \
    CART_PROD_3(ID##78, A, 78, X, Y)       \
    CART_PROD_3(ID##79, A, 79, X, Y)       \
    CART_PROD_3(ID##80, A, 80, X, Y)       \
    CART_PROD_3(ID##81, A, 81, X, Y)       \
    CART_PROD_3(ID##82, A, 82, X, Y)       \
    CART_PROD_3(ID##83, A, 83, X, Y)       \
    CART_PROD_3(ID##84, A, 84, X, Y)       \
    CART_PROD_3(ID##85, A, 85, X, Y)       \
    CART_PROD_3(ID##86, A, 86, X, Y)       \
    CART_PROD_3(ID##87, A, 87, X, Y)       \
    CART_PROD_3(ID##88, A, 88, X, Y)       \
    CART_PROD_3(ID##89, A, 89, X, Y)       \
    CART_PROD_3(ID##90, A, 90, X, Y)       \
    CART_PROD_3(ID##91, A, 91, X, Y)       \
    CART_PROD_3(ID##92, A, 92, X, Y)       \
    CART_PROD_3(ID##93, A, 93, X, Y)       \
    CART_PROD_3(ID##94, A, 94, X, Y)       \
    CART_PROD_3(ID##95, A, 95, X, Y)       \
    CART_PROD_3(ID##96, A, 96, X, Y)       \
    CART_PROD_3(ID##97, A, 97, X, Y)       \
    CART_PROD_3(ID##98, A, 98, X, Y)       \
    CART_PROD_3(ID##99, A, 99, X, Y)       \
    CART_PROD_3(ID##100, A, 100, X, Y)     \
    CART_PROD_3(ID##101, A, 101, X, Y)     \
    CART_PROD_3(ID##102, A, 102, X, Y)     \
    CART_PROD_3(ID##103, A, 103, X, Y)     \
    CART_PROD_3(ID##104, A, 104, X, Y)     \
    CART_PROD_3(ID##105, A, 105, X, Y)     \
    CART_PROD_3(ID##106, A, 106, X, Y)     \
    CART_PROD_3(ID##107, A, 107, X, Y)     \
    CART_PROD_3(ID##108, A, 108, X, Y)     \
    CART_PROD_3(ID##109, A, 109, X, Y)     \
    CART_PROD_3(ID##110, A, 110, X, Y)     \
    CART_PROD_3(ID##111, A, 111, X, Y)     \
    CART_PROD_3(ID##112, A, 112, X, Y)     \
    CART_PROD_3(ID##113, A, 113, X, Y)     \
    CART_PROD_3(ID##114, A, 114, X, Y)     \
    CART_PROD_3(ID##115, A, 115, X, Y)     \
    CART_PROD_3(ID##116, A, 116, X, Y)     \
    CART_PROD_3(ID##117, A, 117, X, Y)     \
    CART_PROD_3(ID##118, A, 118, X, Y)     \
    CART_PROD_3(ID##119, A, 119, X, Y)     \
    CART_PROD_3(ID##120, A, 120, X, Y)     \
    CART_PROD_3(ID##121, A, 121, X, Y)     \
    CART_PROD_3(ID##122, A, 122, X, Y)     \
    CART_PROD_3(ID##123, A, 123, X, Y)     \
    CART_PROD_3(ID##124, A, 124, X, Y)     \
    CART_PROD_3(ID##125, A, 125, X, Y)     \
    CART_PROD_3(ID##126, A, 126, X, Y)     \
    CART_PROD_3(ID##127, A, 127, X, Y)     \
 

#define CART_PROD_2_256(ID, A, X, Y)           \
    CART_PROD_3(ID##1, A, 1, X, Y)             \
    CART_PROD_3(ID##2, A, 2, X, Y)             \
    CART_PROD_3(ID##3, A, 3, X, Y)             \
    CART_PROD_3(ID##4, A, 4, X, Y)             \
    CART_PROD_3(ID##5, A, 5, X, Y)             \
    CART_PROD_3(ID##6, A, 6, X, Y)             \
    CART_PROD_3(ID##7, A, 7, X, Y)             \
    CART_PROD_3(ID##8, A, 8, X, Y)             \
    CART_PROD_3(ID##9, A, 9, X, Y)             \
    CART_PROD_3(ID##10, A, 10, X, Y)           \
    CART_PROD_3(ID##11, A, 11, X, Y)           \
    CART_PROD_3(ID##12, A, 12, X, Y)           \
    CART_PROD_3(ID##13, A, 13, X, Y)           \
    CART_PROD_3(ID##14, A, 14, X, Y)           \
    CART_PROD_3(ID##15, A, 15, X, Y)           \
    CART_PROD_3(ID##16, A, 16, X, Y)           \
    CART_PROD_3(ID##17, A, 17, X, Y)           \
    CART_PROD_3(ID##18, A, 18, X, Y)           \
    CART_PROD_3(ID##19, A, 19, X, Y)           \
    CART_PROD_3(ID##20, A, 20, X, Y)           \
    CART_PROD_3(ID##21, A, 21, X, Y)           \
    CART_PROD_3(ID##22, A, 22, X, Y)           \
    CART_PROD_3(ID##23, A, 23, X, Y)           \
    CART_PROD_3(ID##24, A, 24, X, Y)           \
    CART_PROD_3(ID##25, A, 25, X, Y)           \
    CART_PROD_3(ID##26, A, 26, X, Y)           \
    CART_PROD_3(ID##27, A, 27, X, Y)           \
    CART_PROD_3(ID##28, A, 28, X, Y)           \
    CART_PROD_3(ID##29, A, 29, X, Y)           \
    CART_PROD_3(ID##30, A, 30, X, Y)           \
    CART_PROD_3(ID##31, A, 31, X, Y)           \
    CART_PROD_3(ID##32, A, 32, X, Y)           \
    CART_PROD_3(ID##33, A, 33, X, Y)           \
    CART_PROD_3(ID##34, A, 34, X, Y)           \
    CART_PROD_3(ID##35, A, 35, X, Y)           \
    CART_PROD_3(ID##36, A, 36, X, Y)           \
    CART_PROD_3(ID##37, A, 37, X, Y)           \
    CART_PROD_3(ID##38, A, 38, X, Y)           \
    CART_PROD_3(ID##39, A, 39, X, Y)           \
    CART_PROD_3(ID##40, A, 40, X, Y)           \
    CART_PROD_3(ID##41, A, 41, X, Y)           \
    CART_PROD_3(ID##42, A, 42, X, Y)           \
    CART_PROD_3(ID##43, A, 43, X, Y)           \
    CART_PROD_3(ID##44, A, 44, X, Y)           \
    CART_PROD_3(ID##45, A, 45, X, Y)           \
    CART_PROD_3(ID##46, A, 46, X, Y)           \
    CART_PROD_3(ID##47, A, 47, X, Y)           \
    CART_PROD_3(ID##48, A, 48, X, Y)           \
    CART_PROD_3(ID##49, A, 49, X, Y)           \
    CART_PROD_3(ID##50, A, 50, X, Y)           \
    CART_PROD_3(ID##51, A, 51, X, Y)           \
    CART_PROD_3(ID##52, A, 52, X, Y)           \
    CART_PROD_3(ID##53, A, 53, X, Y)           \
    CART_PROD_3(ID##54, A, 54, X, Y)           \
    CART_PROD_3(ID##55, A, 55, X, Y)           \
    CART_PROD_3(ID##56, A, 56, X, Y)           \
    CART_PROD_3(ID##57, A, 57, X, Y)           \
    CART_PROD_3(ID##58, A, 58, X, Y)           \
    CART_PROD_3(ID##59, A, 59, X, Y)           \
    CART_PROD_3(ID##60, A, 60, X, Y)           \
    CART_PROD_3(ID##61, A, 61, X, Y)           \
    CART_PROD_3(ID##62, A, 62, X, Y)           \
    CART_PROD_3(ID##63, A, 63, X, Y)           \
    CART_PROD_3(ID##64, A, 64, X, Y)           \
    CART_PROD_3(ID##65, A, 65, X, Y)           \
    CART_PROD_3(ID##66, A, 66, X, Y)           \
    CART_PROD_3(ID##67, A, 67, X, Y)           \
    CART_PROD_3(ID##68, A, 68, X, Y)           \
    CART_PROD_3(ID##69, A, 69, X, Y)           \
    CART_PROD_3(ID##70, A, 70, X, Y)           \
    CART_PROD_3(ID##71, A, 71, X, Y)           \
    CART_PROD_3(ID##72, A, 72, X, Y)           \
    CART_PROD_3(ID##73, A, 73, X, Y)           \
    CART_PROD_3(ID##74, A, 74, X, Y)           \
    CART_PROD_3(ID##75, A, 75, X, Y)           \
    CART_PROD_3(ID##76, A, 76, X, Y)           \
    CART_PROD_3(ID##77, A, 77, X, Y)           \
    CART_PROD_3(ID##78, A, 78, X, Y)           \
    CART_PROD_3(ID##79, A, 79, X, Y)           \
    CART_PROD_3(ID##80, A, 80, X, Y)           \
    CART_PROD_3(ID##81, A, 81, X, Y)           \
    CART_PROD_3(ID##82, A, 82, X, Y)           \
    CART_PROD_3(ID##83, A, 83, X, Y)           \
    CART_PROD_3(ID##84, A, 84, X, Y)           \
    CART_PROD_3(ID##85, A, 85, X, Y)           \
    CART_PROD_3(ID##86, A, 86, X, Y)           \
    CART_PROD_3(ID##87, A, 87, X, Y)           \
    CART_PROD_3(ID##88, A, 88, X, Y)           \
    CART_PROD_3(ID##89, A, 89, X, Y)           \
    CART_PROD_3(ID##90, A, 90, X, Y)           \
    CART_PROD_3(ID##91, A, 91, X, Y)           \
    CART_PROD_3(ID##92, A, 92, X, Y)           \
    CART_PROD_3(ID##93, A, 93, X, Y)           \
    CART_PROD_3(ID##94, A, 94, X, Y)           \
    CART_PROD_3(ID##95, A, 95, X, Y)           \
    CART_PROD_3(ID##96, A, 96, X, Y)           \
    CART_PROD_3(ID##97, A, 97, X, Y)           \
    CART_PROD_3(ID##98, A, 98, X, Y)           \
    CART_PROD_3(ID##99, A, 99, X, Y)           \
    CART_PROD_3(ID##100, A, 100, X, Y)         \
    CART_PROD_3(ID##101, A, 101, X, Y)         \
    CART_PROD_3(ID##102, A, 102, X, Y)         \
    CART_PROD_3(ID##103, A, 103, X, Y)         \
    CART_PROD_3(ID##104, A, 104, X, Y)         \
    CART_PROD_3(ID##105, A, 105, X, Y)         \
    CART_PROD_3(ID##106, A, 106, X, Y)         \
    CART_PROD_3(ID##107, A, 107, X, Y)         \
    CART_PROD_3(ID##108, A, 108, X, Y)         \
    CART_PROD_3(ID##109, A, 109, X, Y)         \
    CART_PROD_3(ID##110, A, 110, X, Y)         \
    CART_PROD_3(ID##111, A, 111, X, Y)         \
    CART_PROD_3(ID##112, A, 112, X, Y)         \
    CART_PROD_3(ID##113, A, 113, X, Y)         \
    CART_PROD_3(ID##114, A, 114, X, Y)         \
    CART_PROD_3(ID##115, A, 115, X, Y)         \
    CART_PROD_3(ID##116, A, 116, X, Y)         \
    CART_PROD_3(ID##117, A, 117, X, Y)         \
    CART_PROD_3(ID##118, A, 118, X, Y)         \
    CART_PROD_3(ID##119, A, 119, X, Y)         \
    CART_PROD_3(ID##120, A, 120, X, Y)         \
    CART_PROD_3(ID##121, A, 121, X, Y)         \
    CART_PROD_3(ID##122, A, 122, X, Y)         \
    CART_PROD_3(ID##123, A, 123, X, Y)         \
    CART_PROD_3(ID##124, A, 124, X, Y)         \
    CART_PROD_3(ID##125, A, 125, X, Y)         \
    CART_PROD_3(ID##126, A, 126, X, Y)         \
    CART_PROD_3(ID##127, A, 127, X, Y)         \
    CART_PROD_3(ID##128, A, 128, X, Y)         \
    CART_PROD_3(ID##129, A, 129, X, Y)         \
    CART_PROD_3(ID##130, A, 130, X, Y)         \
    CART_PROD_3(ID##131, A, 131, X, Y)         \
    CART_PROD_3(ID##132, A, 132, X, Y)         \
    CART_PROD_3(ID##133, A, 133, X, Y)         \
    CART_PROD_3(ID##134, A, 134, X, Y)         \
    CART_PROD_3(ID##135, A, 135, X, Y)         \
    CART_PROD_3(ID##136, A, 136, X, Y)         \
    CART_PROD_3(ID##137, A, 137, X, Y)         \
    CART_PROD_3(ID##138, A, 138, X, Y)         \
    CART_PROD_3(ID##139, A, 139, X, Y)         \
    CART_PROD_3(ID##140, A, 140, X, Y)         \
    CART_PROD_3(ID##141, A, 141, X, Y)         \
    CART_PROD_3(ID##142, A, 142, X, Y)         \
    CART_PROD_3(ID##143, A, 143, X, Y)         \
    CART_PROD_3(ID##144, A, 144, X, Y)         \
    CART_PROD_3(ID##145, A, 145, X, Y)         \
    CART_PROD_3(ID##146, A, 146, X, Y)         \
    CART_PROD_3(ID##147, A, 147, X, Y)         \
    CART_PROD_3(ID##148, A, 148, X, Y)         \
    CART_PROD_3(ID##149, A, 149, X, Y)         \
    CART_PROD_3(ID##150, A, 150, X, Y)         \
    CART_PROD_3(ID##151, A, 151, X, Y)         \
    CART_PROD_3(ID##152, A, 152, X, Y)         \
    CART_PROD_3(ID##153, A, 153, X, Y)         \
    CART_PROD_3(ID##154, A, 154, X, Y)         \
    CART_PROD_3(ID##155, A, 155, X, Y)         \
    CART_PROD_3(ID##156, A, 156, X, Y)         \
    CART_PROD_3(ID##157, A, 157, X, Y)         \
    CART_PROD_3(ID##158, A, 158, X, Y)         \
    CART_PROD_3(ID##159, A, 159, X, Y)         \
    CART_PROD_3(ID##160, A, 160, X, Y)         \
    CART_PROD_3(ID##161, A, 161, X, Y)         \
    CART_PROD_3(ID##162, A, 162, X, Y)         \
    CART_PROD_3(ID##163, A, 163, X, Y)         \
    CART_PROD_3(ID##164, A, 164, X, Y)         \
    CART_PROD_3(ID##165, A, 165, X, Y)         \
    CART_PROD_3(ID##166, A, 166, X, Y)         \
    CART_PROD_3(ID##167, A, 167, X, Y)         \
    CART_PROD_3(ID##168, A, 168, X, Y)         \
    CART_PROD_3(ID##169, A, 169, X, Y)         \
    CART_PROD_3(ID##170, A, 170, X, Y)         \
    CART_PROD_3(ID##171, A, 171, X, Y)         \
    CART_PROD_3(ID##172, A, 172, X, Y)         \
    CART_PROD_3(ID##173, A, 173, X, Y)         \
    CART_PROD_3(ID##174, A, 174, X, Y)         \
    CART_PROD_3(ID##175, A, 175, X, Y)         \
    CART_PROD_3(ID##176, A, 176, X, Y)         \
    CART_PROD_3(ID##177, A, 177, X, Y)         \
    CART_PROD_3(ID##178, A, 178, X, Y)         \
    CART_PROD_3(ID##179, A, 179, X, Y)         \
    CART_PROD_3(ID##180, A, 180, X, Y)         \
    CART_PROD_3(ID##181, A, 181, X, Y)         \
    CART_PROD_3(ID##182, A, 182, X, Y)         \
    CART_PROD_3(ID##183, A, 183, X, Y)         \
    CART_PROD_3(ID##184, A, 184, X, Y)         \
    CART_PROD_3(ID##185, A, 185, X, Y)         \
    CART_PROD_3(ID##186, A, 186, X, Y)         \
    CART_PROD_3(ID##187, A, 187, X, Y)         \
    CART_PROD_3(ID##188, A, 188, X, Y)         \
    CART_PROD_3(ID##189, A, 189, X, Y)         \
    CART_PROD_3(ID##190, A, 190, X, Y)         \
    CART_PROD_3(ID##191, A, 191, X, Y)         \
    CART_PROD_3(ID##192, A, 192, X, Y)         \
    CART_PROD_3(ID##193, A, 193, X, Y)         \
    CART_PROD_3(ID##194, A, 194, X, Y)         \
    CART_PROD_3(ID##195, A, 195, X, Y)         \
    CART_PROD_3(ID##196, A, 196, X, Y)         \
    CART_PROD_3(ID##197, A, 197, X, Y)         \
    CART_PROD_3(ID##198, A, 198, X, Y)         \
    CART_PROD_3(ID##199, A, 199, X, Y)         \
    CART_PROD_3(ID##200, A, 200, X, Y)         \
    CART_PROD_3(ID##201, A, 201, X, Y)         \
    CART_PROD_3(ID##202, A, 202, X, Y)         \
    CART_PROD_3(ID##203, A, 203, X, Y)         \
    CART_PROD_3(ID##204, A, 204, X, Y)         \
    CART_PROD_3(ID##205, A, 205, X, Y)         \
    CART_PROD_3(ID##206, A, 206, X, Y)         \
    CART_PROD_3(ID##207, A, 207, X, Y)         \
    CART_PROD_3(ID##208, A, 208, X, Y)         \
    CART_PROD_3(ID##209, A, 209, X, Y)         \
    CART_PROD_3(ID##210, A, 210, X, Y)         \
    CART_PROD_3(ID##211, A, 211, X, Y)         \
    CART_PROD_3(ID##212, A, 212, X, Y)         \
    CART_PROD_3(ID##213, A, 213, X, Y)         \
    CART_PROD_3(ID##214, A, 214, X, Y)         \
    CART_PROD_3(ID##215, A, 215, X, Y)         \
    CART_PROD_3(ID##216, A, 216, X, Y)         \
    CART_PROD_3(ID##217, A, 217, X, Y)         \
    CART_PROD_3(ID##218, A, 218, X, Y)         \
    CART_PROD_3(ID##219, A, 219, X, Y)         \
    CART_PROD_3(ID##220, A, 220, X, Y)         \
    CART_PROD_3(ID##221, A, 221, X, Y)         \
    CART_PROD_3(ID##222, A, 222, X, Y)         \
    CART_PROD_3(ID##223, A, 223, X, Y)         \
    CART_PROD_3(ID##224, A, 224, X, Y)         \
    CART_PROD_3(ID##225, A, 225, X, Y)         \
    CART_PROD_3(ID##226, A, 226, X, Y)         \
    CART_PROD_3(ID##227, A, 227, X, Y)         \
    CART_PROD_3(ID##228, A, 228, X, Y)         \
    CART_PROD_3(ID##229, A, 229, X, Y)         \
    CART_PROD_3(ID##230, A, 230, X, Y)         \
    CART_PROD_3(ID##231, A, 231, X, Y)         \
    CART_PROD_3(ID##232, A, 232, X, Y)         \
    CART_PROD_3(ID##233, A, 233, X, Y)         \
    CART_PROD_3(ID##234, A, 234, X, Y)         \
    CART_PROD_3(ID##235, A, 235, X, Y)         \
    CART_PROD_3(ID##236, A, 236, X, Y)         \
    CART_PROD_3(ID##237, A, 237, X, Y)         \
    CART_PROD_3(ID##238, A, 238, X, Y)         \
    CART_PROD_3(ID##239, A, 239, X, Y)         \
    CART_PROD_3(ID##240, A, 240, X, Y)         \
    CART_PROD_3(ID##241, A, 241, X, Y)         \
    CART_PROD_3(ID##242, A, 242, X, Y)         \
    CART_PROD_3(ID##243, A, 243, X, Y)         \
    CART_PROD_3(ID##244, A, 244, X, Y)         \
    CART_PROD_3(ID##245, A, 245, X, Y)         \
    CART_PROD_3(ID##246, A, 246, X, Y)         \
    CART_PROD_3(ID##247, A, 247, X, Y)         \
    CART_PROD_3(ID##248, A, 248, X, Y)         \
    CART_PROD_3(ID##249, A, 249, X, Y)         \
    CART_PROD_3(ID##250, A, 250, X, Y)         \


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
