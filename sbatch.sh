#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=benchmark.json

./build/bin/pricing_cli benchmark --filter-by-name vanilla_american_binomial_cuda_bkdstprcmp_xdovlpunroll_vtile_trimotm --parameters large-reprare --reference-function vanilla_american_binomial_cuda_naive --output-format=json
