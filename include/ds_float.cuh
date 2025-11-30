#pragma once

#include <cuda_runtime.h>

// Double-Single (DS) arithmetic: Emulate ~48-bit mantissa precision using pairs of FP32
// Each DS number is represented as (x, y) where value ≈ x + y (x=high-order, y=low-order)
// This provides precision between FP32 (24 bits) and FP64 (53 bits)

/// IMPLEMENTATION of the paper: Extended-Precision Floating-Point Numbers for GPU Computation https://andrewthall.org/papers/df64_qf128.pdf

// Use CUDA's native float2 type for optimal memory alignment and vectorization
// Note: float2 uses .x (high-order) and .y (low-order) field naming
typedef float2 ds_float;

// ============================================================================
// Conversion Functions
// ============================================================================

__device__ __host__ inline ds_float double_to_ds(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return make_float2(hi, lo);
}

__device__ __host__ inline double ds_to_double(ds_float x) {
    return (double)x.x + (double)x.y;
}

__device__ __host__ inline ds_float float_to_ds(float x) {
    return make_float2(x, 0.0f);
}

// ============================================================================
// Basic Arithmetic Helper Functions (from Thall paper)
// ============================================================================

// Error-free transformation functions return ds_float where:
// - .x contains the primary result (sum or product)
// - .y contains the rounding error correction

__device__ __host__ inline ds_float quickTwoSum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return make_float2(s, e);
}

__device__ __host__ inline ds_float twoSum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return make_float2(s, e);
}

// split_result is kept separate as it represents bit-level decomposition,
// not a value+error representation
struct split_result {
    float hi;
    float lo;
};

__device__ __host__ inline split_result split(float a) {
    const float SPLIT = 4097.0f;  // (1 << 12) + 1
    float t = a * SPLIT;
    float ahi = t - (t - a);
    float alo = a - ahi;
    return split_result{ahi, alo};
}

__device__ __host__ inline ds_float twoProd(float a, float b) {
    float p = a * b;
    split_result aS = split(a);
    split_result bS = split(b);
    float err = ((aS.hi * bS.hi - p)
                 + aS.hi * bS.lo + aS.lo * bS.hi)
                 + aS.lo * bS.lo;
    return make_float2(p, err);
}

__device__ inline ds_float twoProdFMA(float a, float b) {
    float p = a * b;
    float err = __fmaf_rn(a, b, -p);
    return make_float2(p, err);
}

// ============================================================================
// Basic Arithmetic Operations
// ============================================================================

__device__ inline ds_float ds_add(ds_float a, ds_float b) {
    ds_float s = twoSum(a.x, b.x);

    ds_float t = twoSum(a.y, b.y);

    s.y += t.x;

    ds_float r1 = quickTwoSum(s.x, s.y);

    r1.y += t.y;

    ds_float r2 = quickTwoSum(r1.x, r1.y);

    return make_float2(r2.x, r2.y);
}

//Optimized version
__device__ inline ds_float ds_add_opt(ds_float a, ds_float b) {
    float2 s = make_float2(a.x + b.x, a.y + b.y);
    float2 v = make_float2(s.x - a.x, s.y - a.y);
    float2 e = make_float2((a.x - (s.x - v.x)) + (b.x - v.x),
                           (a.y - (s.y - v.y)) + (b.y - v.y));

    e.x += s.y;

    ds_float r1 = quickTwoSum(s.x, e.x);

    r1.y += e.y;

    ds_float r2 = quickTwoSum(r1.x, r1.y);

    return make_float2(r2.x, r2.y);
}

__device__ inline ds_float ds_mul(ds_float a, ds_float b) {
    ds_float p = twoProd(a.x, b.x);

    ds_float t1 = twoSum(p.y, a.x * b.y);
    ds_float t2 = twoSum(t1.x, a.y * b.x);
    ds_float t3 = twoSum(t2.x, a.y * b.y);

    float accumulated_err = t1.y + t2.y + t3.y;

    ds_float result = twoSum(p.x, t3.x);

    result.y += accumulated_err;

    return make_float2(result.x, result.y);
}

//Optimized version
__device__ inline ds_float ds_mul_opt(ds_float a, ds_float b) {
    ds_float p = twoProdFMA(a.x, b.x);

    float err = p.y + (a.x * b.y + a.y * b.x);

    ds_float result = quickTwoSum(p.x, err);

    return make_float2(result.x, result.y);
}

__device__ inline ds_float ds_normalize(ds_float x) {
    ds_float r = twoSum(x.x, x.y);
    return make_float2(r.x, r.y);
}

// ============================================================================
// Comparison Operations
// ============================================================================

__device__ inline bool ds_less(ds_float a, ds_float b) {
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return a.y < b.y;
}

__device__ inline bool ds_greater(ds_float a, ds_float b) {
    if (a.x > b.x) return true;
    if (a.x < b.x) return false;
    return a.y > b.y;
}

__device__ inline ds_float ds_max(ds_float a, ds_float b) {
    return ds_greater(a, b) ? a : b;
}

__device__ inline ds_float ds_min(ds_float a, ds_float b) {
    return ds_less(a, b) ? a : b;
}

// ============================================================================
// Warp Shuffle Operations
// ============================================================================

__device__ inline ds_float ds_shfl_down_sync(unsigned mask, ds_float val, int delta) {
    float hi = __shfl_down_sync(mask, val.x, delta);
    float lo = __shfl_down_sync(mask, val.y, delta);
    return make_float2(hi, lo);
}