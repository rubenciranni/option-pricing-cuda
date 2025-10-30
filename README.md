# Option Pricing CUDA

## Development
## Install Pre Commit Hooks
Install pre-commit either with pip or brew, then run:
```
pre-commit install
```

### Build
```bash
mkdir build
cd build
cmake -DCMAKE_CUDA_COMPILER=/cluster/data/cuda/13.0.0/bin/nvcc ..
make
```

### Implement a new backend for an existing model
- Existing models are inside `src/backends/models` for example `vanilla_american_binomial.cpp`
- Place your new implementation inside `src/backends/<backend>/<model>_<backend>.cpp`
- Place the corresponding function declaration inside `include/backends/<backend>/<model>_<backend>.hpp`
- Add your backend to `benchmarks/functions_version.cpp` (for benchmarking)

## CLI Tool Usage
### Price an option
- run `cd build/bin`
- run `./pricing_cli price --help` to learn about parameters.
- Example: `./pricing_cli price -S 100 -K 100 -T 1 -r 0.03 -q 0.015 -n 1000`.

### Run benchmarks
- Choose one of the benchmarks parameters from `benchmarks/benchmark_parameters.cpp`, or add a new one.
- build
- run `cd build/bin`
- run ``./pricing_cli benchmark --help` to learn about parameters.
- Example: `./pricing_cli benchmark --filter-by-name vanilla_american_binomial_cpu --parameters easy`.

## Testing

```bash
cd build/tests
ctest -V
```



# GPU command
```bash
srun -A dphpc -t 60:00 --gpus 5060ti:1 --pty bash
```