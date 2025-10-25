#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=benchmark.out


./build/bin/pricing_cli benchmark
