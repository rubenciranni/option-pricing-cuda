#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=benchmark.json

./build/bin/pricing_cli benchmark --filter-by-name vanilla_american_binomial_cuda_overlap_unroll --parameters hard --reference-function vanilla_american_binomial_cuda_no_sync --output-format=json
