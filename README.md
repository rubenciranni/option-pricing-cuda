# Option Pricing CUDA

## CLI Tool Usage
TODO

## Development

### Structure

```
.
option_pricing/
├── CMakeLists.txt                  # Root CMake configuration
├── README.md
│
├── cmake/                          # Custom CMake modules and build scripts
│
├── include/                        # Public headers (exported API)
│   └── option_pricing/
│       ├── models/                 # High-level pricing model interfaces (wrappers to backends)
│       │   ├── vanilla_european_binomial/           # Each model has its own folder
│       │   ├── vanilla_american_binomial/
│       │   └── ...
│       │
│       └── backends/               # Backend interface headers
│           ├── cpu/
│           │   ├── vanilla_european_binomial/
│           │   └── ...
│           ├── cuda/
│           │   ├── vanilla_european_binomial/
│           │   └── ...
│           └── openmp/
│               ├── vanilla_european_binomial/
│               └── ...
│
├── src/                            # Library implementation files
│   └── option_pricing/             # Mirrors include/option_pricing
│       ├── pricing_models/         # High-level pricing model interfaces (wrappers to backends)
│       │   ├── vanilla_european_binomial/
│       │   ├── vanilla_american_binomial/
│       │   └── ...
│       │
│       └── backends/               # Backend-specific implementations
│           ├── cpu/
│           │   ├── vanilla_european_binomial/
│           │   └── ...
│           ├── cuda/
│           │   ├── vanilla_european_binomial/
│           │   └── ...
│           └── openmp/
│               ├── vanilla_european_binomial/
│               └── ...
│
├── apps/                           # CLI tools and example executables
│   ├── include/                    # App-specific headers (CLI helpers, argument parsing)
│   └── ...
│
├── benchmarks/                     # Performance benchmarks
│   ├── include/                    # Benchmark utilities and datasets
│   └── ...
│
├── tests/                          # Unit and integration tests
│   ├── include/                    # Test utilities, fixtures, shared data
│   └── ...
│
├── internal/                       # Shared internal-only utilities
│   └── include/
│
├── scripts/                        # Development/debug/data-prep scripts
│
└── build/                          # Out-of-source build directory (ignored in Git)
```

### Building

```bash
mkdir build
cd build
cmake ..
make
```