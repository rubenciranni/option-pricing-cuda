#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --account=dphpc
#SBATCH --output=a.json

./build/bin/pricing_cli benchmark --filter-by-name vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds --parameters xl-125   --reference-function vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm --output-format=json > 512-ds-NStep125-100.json
./build/bin/pricing_cli benchmark --filter-by-name vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds --parameters xl-crazy --reference-function vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm --output-format=json > 512-ds-NStepCM2-111.json
