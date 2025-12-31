#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --account=dphpc
#SBATCH --output=final-o3.json

./build/bin/pricing_cli benchmark --filter-by-name vanilla --parameters l-125 --reference-function vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_shuffle_trimotm_ds --output-format=json
