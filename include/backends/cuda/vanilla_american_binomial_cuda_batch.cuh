#pragma once
#include <vector>

#include "backends/hyperparams.hpp"
#include "constants.hpp"

// Batch CUDA kernel declarations
// These kernels process multiple pricing inputs in a single call

void vanilla_american_binomial_cuda_batch_naive(std::vector<PricingInput>& runs,
                                                std::vector<double>& out);

void vanilla_american_binomial_cuda_batch_nvidia_baseline(std::vector<PricingInput>& runs,
                                                          std::vector<double>& out);

void vanilla_american_binomial_cuda_batch_stprcmp(std::vector<PricingInput>& runs,
                                                  std::vector<double>& out);

void vanilla_american_binomial_cuda_batch_bkdstprcmp(std::vector<PricingInput>& runs,
                                                     std::vector<double>& out);

template <const Hyperparams& h>
void vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_vtile_trimotm(
    std::vector<PricingInput>& runs, std::vector<double>& out);

template <const Hyperparams& h>
void vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds(
    std::vector<PricingInput>& runs, std::vector<double>& out);

template <const Hyperparams& h>
void test_vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds(
    std::vector<PricingInput>& runs, std::vector<double>& out);

template <const Hyperparams& h>
void vanilla_american_binomial_cuda_batch_bkdstprcmp_xdovlpunroll_shuffle_trimotm(
    std::vector<PricingInput>& runs, std::vector<double>& out);

void vanilla_american_binomial_cuda_batch_scheduler_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds(
    std::vector<PricingInput>& runs, std::vector<double>& out);
