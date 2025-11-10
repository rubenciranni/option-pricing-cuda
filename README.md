# Option Pricing CUDA

## Development
### Install Pre Commit Hooks
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
- Add your backend to the function registry `benchmarks/benchmark.cpp` (for benchmarking)

### Hyperparameters

All backends (cpu/cuda) can be defined as template functions parametrized by a `Hyperparams` class instance defined in `include/backends/hyperparams.hpp`. 

> The `Hyperparams` class can always be modified to fit the ever-expanding needs of this project. Need another hyperparameter? Add a new attribute to the `Hyperparams` class and to its constructor, or if you wish you can create another construtor. You decide. Once the class is well-structured, define multiple istances of the `Hyperparams` class, still in the `hyperparams.hpp` file, each containing one parameter configuration you would like to experiment with.

Once the file `hyperparams.hpp` is well-stuctured, do the following:
* include in your file `#include "backends/hyperparams.hpp"`
* Add `template <const Hyperparams& h>` to the top of your backend. Then parametrize the cuda kernels. Please keep the convention that the cuda kernerls are parametrized only with the parameters they really need. For example, add `template<const int UNROLL_FACTOR>` to the top of your kernel.
* Add `template <const Hyperparams& h>` to the top of your backend in the `include/backends/cuda/vanilla_american_binomial_cuda.cuh` or your corresponding file in `include/backends`. 
* Add your function to the function registry in `benchmarks/benchmark.cpp` as many times as you want, as follows `{"func_H1", func<H1>},...{"func_H9", func<H9>}`.

Everything should now work.

### Grid search

One additional functionality that this project has is the possibility of doing a grid-search at compile time of the 

`include/backends/hyperparams.hpp`. 
The following macro will define a grid search of possible Hyperparams instances based on the hyperparameters you have added in the CART_PROD_i macros. Each instance will be called GRID_SEARCH_HYPERPARAMS_ID, with ID,  

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


