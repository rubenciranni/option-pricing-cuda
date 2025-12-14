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

// Assuming ds_float is float2 and helper functions are defined

__device__ inline ds_float ds_add_two_mults_streamlined(
    ds_float my_up, ds_float up_val, 
    ds_float my_down, ds_float val) 
{
    // --- Phase 1: High-Part Calculation (Maximizing FMA) ---
    // Calculate the two high-part products: P_AB_h = my_up.x * up_val.x
    // and P_CD_h = my_down.x * val.x.
    
    // We can use the Two-Product logic here, ensuring we generate the high-part (p) 
    // and the high-part error (err_AB_h), and then sum the two high parts.
    
    // P_AB = my_up.x * up_val.x + error
    float p_AB = my_up.x * up_val.x;
    float err_AB_h = __fmaf_rn(my_up.x, up_val.x, -p_AB);
    
    // P_CD = my_down.x * val.x + error
    float p_CD = my_down.x * val.x;
    float err_CD_h = __fmaf_rn(my_down.x, val.x, -p_CD);

    // Initial sum of the two high parts (p_AB + p_CD) using quickTwoSum
    // This is the most crucial addition step.
    ds_float s_high = quickTwoSum(p_AB, p_CD); // R_h = s_high.x, R_e1 = s_high.y

    // --- Phase 2: Error Accumulation (The Low Parts) ---
    // Calculate all the remaining low-order cross terms and the high-part errors.

    float low_sum = s_high.y; // Start the low sum with the error from the high-part addition (R_e1)

    // Add the error from the two high-part products (err_AB_h and err_CD_h)
    low_sum += err_AB_h;
    low_sum += err_CD_h;
    
    // Add the four cross-terms, explicitly using FMA where possible for efficiency
    // Term 1: my_up.x * up_val.y
    low_sum = __fmaf_rn(my_up.x, up_val.y, low_sum); 
    
    // Term 2: my_up.y * up_val.x
    low_sum = __fmaf_rn(my_up.y, up_val.x, low_sum);
    
    // Term 3: my_down.x * val.y
    low_sum = __fmaf_rn(my_down.x, val.y, low_sum);
    
    // Term 4: my_down.y * val.x
    low_sum = __fmaf_rn(my_down.y, val.x, low_sum);

    // --- Phase 3: Final Consolidation (quickTwoSum) ---
    // Combine the high part (s_high.x) with the accumulated low sum (low_sum)
    // The final result is refined using a high-precision addition.
    ds_float final_result = quickTwoSum(s_high.x, low_sum); 

    return final_result;
}

__device__ inline ds_float ds_mul_opt_fused(ds_float a, ds_float b) {
    ds_float p = twoProdFMA(a.x, b.x); // Still uses 2 instructions

    // Explicitly enforce FMA for the cross-term additions
    // err = p.y + (a.x * b.y) + (a.y * b.x);
    
    // Use FMA for the first term: (a.x * b.y) + p.y
    float temp_err = __fmaf_rn(a.x, b.y, p.y); 
    
    // Add the remaining term: temp_err + (a.y * b.x)
    // We can't guarantee a second FMA, so we rely on compiler for the last step
    float final_err = __fmaf_rn(a.y, b.x, temp_err); // This ensures two FMA uses!

    ds_float result = quickTwoSum(p.x, final_err);

    return make_float2(result.x, result.y);
}
__device__ inline ds_float ds_add_two_mults_opt(ds_float my_up, ds_float up_val, ds_float my_down, ds_float val) {
    
    // --- Step 1: Compute P1 = my_up * up_val (using twoProdFMA logic) ---
    // The multiplication of the high parts: P1.x * P1.y
    ds_float p1 = twoProdFMA(my_up.x, up_val.x); 
    // The error of P1 is p1.y + (my_up.x * up_val.y + my_up.y * up_val.x)
    float err1 = p1.y + (my_up.x * up_val.y + my_up.y * up_val.x);
    // Final result of the first multiplication: hold1.x, hold1.y
    ds_float hold1 = quickTwoSum(p1.x, err1); // Equivalent to ds_mul_opt(my_up, up_val)
    
    // --- Step 2: Compute P2 = my_down * val (using twoProdFMA logic) ---
    // The multiplication of the high parts: P2.x * P2.y
    ds_float p2 = twoProdFMA(my_down.x, val.x); 
    // The error of P2 is p2.y + (my_down.x * val.y + my_down.y * val.x)
    float err2 = p2.y + (my_down.x * val.y + my_down.y * val.x);
    // Final result of the second multiplication: hold2.x, hold2.y
    ds_float hold2 = quickTwoSum(p2.x, err2); // Equivalent to ds_mul_opt(my_down, val)

    // --- Step 3: Compute hold1 + hold2 (Equivalent to ds_add_opt(hold1, hold2)) ---
    
    // Original ds_add_opt starts here with hold1 as 'a' and hold2 as 'b'
    float2 s = make_float2(hold1.x + hold2.x, hold1.y + hold2.y);
    float2 v = make_float2(s.x - hold1.x, s.y - hold1.y);
    float2 e = make_float2((hold1.x - (s.x - v.x)) + (hold2.x - v.x),
                           (hold1.y - (s.y - v.y)) + (hold2.y - v.y));

    // This line is often the source of optimization/complexity in DS math
    e.x += s.y; 

    ds_float r1 = quickTwoSum(s.x, e.x);

    r1.y += e.y;

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