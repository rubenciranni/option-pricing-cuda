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


**Here we explain how to make your backend work with hyperparameters.** Go to the file where you defined your backend. do the following:
1. include in your file `#include "backends/hyperparams.hpp"`
2. Add `template <const Hyperparams& h>` to the top of your backend. Then parametrize the cuda kernels. Please keep the convention that the cuda kernerls are parametrized only with the parameters they really need. For example, add `template<const int UNROLL_FACTOR>` to the top of your kernel.
3. Instantiate your function with the hyperparameters you wish to use. for example, add to the bottom of your backend file: 
```c++
template double vanilla_american_binomial_cuda_unroll_tile<DEFAULT_HYPERPARAMS_CUDA_UNROLL_TILE>(
    const double S, const double K, const double T, const double r, const double sigma,
    const double q, const int n, const OptionType type);
```
4. Now go to `include/backends/cuda/vanilla_american_binomial_cuda.cuh`. Here, add `template <const Hyperparams& h>` to the top of your backend. If you did not implement a cuda backend, find your backend in a file in `include/backends`. 
5. Now go to `benchmarks/benchmark.cpp`. Here, add your function to the function registry as many times as you want, as follows `{"func_H1", func<H1>},...{"func_H9", func<H9>}`.

Everything should now work.

### Grid Search

The Hyperparams class that we have analyzed in the previous section is made to effectively and rapidly create more instances of the same backend to test multiple hyperparameter combinations. In this section, we take this idea to its limit, by defining a procedure that **at compile time**, produces exponentially many instances of a template class, to make testing as easy as possible. Since we are working at compile time, we must make use of Macros and, in particular, X macros. X Macros are a powerful tool that C++ employs to reduce dependencies among files. In this section, we are not going to explore how X Macros work, but I'd be happy to explain them to you if you really want to know (contact Luigi). Basically, they are macros that takes parameters.

To begin our analysis, let us split our problem ("implement a grid search method for our prooject") into simpler subproblems. These subproblems are:
1. **Define hyperparams instances.** First, we need to define instances of the Hyperparams class. These instances will be used to parametrize our backends.
2. **Define backend instances.** Once we have defined the Hyperparams instances, we need to generate at compile time instances of our backend that are parametrized by these hyperparameters. Unfortunately this cannot be done directly in the function registry for reasons I will not get into here. Thus, we will need to define multiple backend instances directly in the files where the backends are defined.
3. **Add backend instances to function registry for testing.** The previous step ensures that the functions are built at compile time. Now, we need to ensure that these functions are run when we call the `pricing_cli` app. By the structure of our project this happens only when the functions are added to the function registry.

Let us now analyze this step by step.

#### Define hyperparams instances

Now, having understood the task we set out to solve, let us tackle task (1). To define all hyperparameter instances, we will need to generate the cartesian product of all hyperparameter instances we want to test.

**Example:** Suppose we want to test THREADS_PER_BLOCK in [128, 256], UNROLL_FACTOR in [37, 61], OUTPUTS_PER_THREAD in [4]. We need to generate the following strings:
```c++
inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_00(256, 37, 4); 
inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_01(256, 61, 4); 
inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_10(128, 37, 4); 
inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_11(128, 61, 4);
```

This is done automatically by the following X macros:
```c++
#define CART_PROD_1(X, Y)                 \
    CART_PROD_2(0, 256, X, Y)             \
    CART_PROD_2(1, 128, X, Y)             \

#define CART_PROD_2(ID, A, X, Y)          \
    CART_PROD_3(ID##0, A, 37, X, Y)       \
    CART_PROD_3(ID##1, A, 61, X, Y)       \

#define CART_PROD_3(ID, A, B, X, Y)       \
    CART_PROD_4(ID/*##0*/, A, B, 4, X, Y)

#define CART_PROD_4(ID, A, B, C, X, Y)    \
    CART_PROD_5(ID/*##0*/, A, B, C, 0, X, Y)

#define CART_PROD_5(ID, A, B, C, D, X, Y) \
    X(ID/*##0*/, A, B, C, D, 0, Y)

#define STR(x) #x
#define APPLY_FUNCTION(FUNC, PROD, FUNC_PARAM) PROD(FUNC, FUNC_PARAM)

#define PRODUCE_HYPERPARAMS_INSTANCES_3(ID, A, B, C, D, E, Y) \
    inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_##ID(A, B, C);

APPLY_FUNCTION(PRODUCE_HYPERPARAMS_INSTANCES_3, HYPERPARAMS_CART_PRODUCT, NULL)

```

So let me break this down in the only compents you need to understand to implement your own grid search:
1. Notice that we need to give a different name to each hyperparameter instance (otherwise the compiler is going to complain).
2. The list of hyperparameters that are going to go into the first attribute position is inserted in `CART_PROD_1`. You are free to create as many lines as you want. The 0 to the left of 256 and the 1 to the left of 128 define the ID associated to that value. This ID needs to be unique.
3. The list of hyperparameters that are going to go into the second attribute position is inserted in `CART_PROD_2`. You are free to create as many lines as you want. The 0 to the left of 37 and the 1 to the left of 61 define the ID associated to that value. This ID needs to be such that when concatenated to the IDs of `CART_PROD_1`, they generate unique values.
4. The list of hyperparameters that are going to go into the third attribute position is inserted in `CART_PROD_3`. You are free to create as many lines as you want. Here, we have commented out the ID addition part, as there is only one choice and that is 4.
5. `CART_PROD_4` and `CART_PROD_5` are implemented, if you ever need more hyperparameters, but do not do anything at the moment.
6. `APPLY_FUNCTION(PRODUCE_HYPERPARAMS_INSTANCES_3, HYPERPARAMS_CART_PRODUCT, NULL)` finally generates as many strings `inline constexpr Hyperparams GRID_SEARCH_HYPERPARAMS_{ID}({A}, {B}, {C});` as needed in the grid search, where ID is the ID of the hyperparam instance, A is the first hyperparam, B is the second, C is the third.
7. If you ever need 4 hyperparameters, in the constructor of the hyperparams class, just add the IDs and wanted hyperparams to `CART_PROD_4`. Then, do `APPLY_FUNCTION(PRODUCE_HYPERPARAMS_INSTANCES_4, HYPERPARAMS_CART_PRODUCT, NULL)` where we switched `PRODUCE_HYPERPARAMS_INSTANCES_4` to `PRODUCE_HYPERPARAMS_INSTANCES_3`.

#### Stop Here if your backend is already setup

You can stop reading here if you just wanted to run grid search and grid search was already implemented in your backend. To check if grid search was implemented in your backend, go to `include/backends/hyperparams.hpp` and see if there is a commented line like:
```c++
// #define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_<name of your backend>
```

If yes, you're in luck! Just uncomment the following 2 lines and grid search will run!
```c++
#define DO_CARTESIAN_PRODUCT
#define DO_CARTESIAN_PRODUCT_OF_VANILLA_AMERICAN_CUDA_<name of your backend>
```

#### What to do if your backend is not already setup

If your backend is not already setup, you need to: (2) **Define backend instances**;
(3) **Add backend instances to function registry for testing.**

#### Define backend instances

Go to the file where your backend is defined and paste this lines in its bottom part:

```c++
#ifdef DO_CARTESIAN_PRODUCT 
#ifdef DO_CARTESIAN_PRODUCT_OF_MY_BACKEND
    
    #define PRODUCE_INSTANCES_OF_MY_BACKEND(ID, A, B, C, D, E, Y) template double vanilla_american_binomial_cuda_MY_BACKEND<GRID_SEARCH_HYPERPARAMS_##ID>(const double S, const double K, const double T, const double r, const double sigma, const double q, const int n, const OptionType type);
    APPLY_FUNCTION(PRODUCE_INSTANCES_OF_MY_BACKEND, HYPERPARAMS_CART_PRODUCT, NULL)

#endif
#endif
```

Please modify `MY_BACKEND` with the actual name of your backend

It is a best practise to be able to choose if you have to compile all backends at compile time. This is why we have `DO_CARTESIAN_PRODUCT`, `DO_CARTESIAN_PRODUCT_OF_MY_BACKEND`, where:
1. `DO_CARTESIAN_PRODUCT` if undefined makes it so that nothing relative to hyperparameters is compiled. If defined, this makes it so that all hyperparameter instances `GRID_SEARCH_HYPERPARAMS_ID` are compiled. if you want to define `DO_CARTESIAN_PRODUCT_OF_MY_BACKEND` uncomment the line for it in `include/backends/hyperparams.hpp` 
2.  If `DO_CARTESIAN_PRODUCT_OF_MY_BACKEND` is defined, your backend will be compiled with grid-search. If undefined, it will not. Please, if you want to define `DO_CARTESIAN_PRODUCT_OF_MY_BACKEND` uncomment/create the line for it in `include/backends/hyperparams.hpp` 


#### Add backend instances to function registry for testing

Go to `benchmarks/benchmark.cpp`. Add the following lines inside the function registry:

```c++
#ifdef DO_CARTESIAN_PRODUCT
    #ifdef DO_CARTESIAN_PRODUCT_OF_MY_BACKEND           
        APPLY_FUNCTION(PRODUCE_FUNCTIONS_FOR_REGISTRY, HYPERPARAMS_CART_PRODUCT, vanilla_american_binomial_cuda_MY_BACKEND)             
    #endif
#endif
```
Please modify `MY_BACKEND` with the actual name of your backend.

At this point you are done, everything should work neatly.


### Json Output

The `pricing_cli.cpp` application can output results in json for simpler analysis. Just do:

```bash
srun -A dphpc  bin/pricing_cli benchmark [...] --output-format=json
```




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

## Profiling
Example:
```bash
bash scripts/profiling.sh ".*compute_.*" "bkdstprcmp_xovlpunroll_vtile_trimotm" "l-repeat"
```
The first argument is the kernels to profile (ncu). The second argument is the function to profile (nsys). The last argument is the level.

Example with all Kernels:
```bash
bash scripts/profiling.sh ".*" "bkdstprcmp_xovlpunroll_vtile_trimotm" "l-repeat"
```


# GPU command
```bash
srun -A dphpc -t 60:00 --gpus 5060ti:1 --pty bash
```


