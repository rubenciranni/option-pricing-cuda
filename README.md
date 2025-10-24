# Option Pricing CUDA

A high-performance C++ library and CLI tool for pricing financial options using various computational backends (CPU, OpenMP, CUDA). This project implements the Cox-Ross-Rubinstein (CRR) binomial model for both American and European vanilla options.

## Features

- **Multiple Option Types**: Support for both American and European vanilla options (Call and Put)
- **Multiple Backends**: Choose between CPU, OpenMP (multi-threaded), and CUDA (GPU-accelerated) implementations
- **Binomial Pricing Model**: Implements the Cox-Ross-Rubinstein (CRR) binomial tree method
- **CLI Tool**: Command-line interface for pricing options and running benchmarks
- **Comprehensive Testing**: Unit tests using Catch2 framework
- **Benchmarking Suite**: Performance comparison across different backends and configurations
- **Modern C++17**: Clean, modern C++ codebase with proper abstractions

## Project Structure

```
option-pricing-cuda/
├── include/              # Public header files
│   ├── backends/         # Backend-specific headers (CPU, OpenMP, CUDA)
│   ├── models/           # Option pricing model headers
│   └── constants.hpp     # Common enums and type definitions
├── src/                  # Implementation files
│   ├── backends/         # Backend implementations
│   │   ├── cpu/          # CPU single-threaded implementation
│   │   ├── openmp/       # OpenMP multi-threaded implementation
│   │   └── cuda/         # CUDA GPU implementation
│   └── models/           # Model implementations
├── apps/                 # Application executables
│   └── pricing_cli.cpp   # Command-line interface
├── tests/                # Unit tests
├── benchmarks/           # Performance benchmarks
└── CMakeLists.txt        # Build configuration
```

## Installation

### Prerequisites

- **C++ Compiler**: GCC, Clang, or MSVC with C++17 support
- **CMake**: Version 3.22 or higher
- **CUDA Toolkit** (optional): For GPU-accelerated computations
- **OpenMP** (optional): For multi-threaded CPU computations

### Building from Source

```bash
# Clone the repository
git clone https://github.com/rubenciranni/option-pricing-cuda.git
cd option-pricing-cuda

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

# Run tests (optional)
ctest
```

### Using Nix (Optional)

A `shell.nix` file is provided for reproducible development environments:

```bash
nix-shell
```

## CLI Tool Usage

The CLI tool provides three main commands: `price`, `benchmark`, and `parameters`.

### Pricing a Single Option

```bash
./pricing_cli price \
  --type call \
  --exercise american \
  --method binomial \
  -S 100.0 \
  -K 100.0 \
  -T 0.5 \
  -r 0.03 \
  --sigma 0.2 \
  -q 0.015 \
  -n 60 \
  --backend cpu
```

**Parameters:**
- `--type`: Option type (`call` or `put`)
- `--exercise`: Exercise style (`american` or `european`)
- `--method`: Pricing method (`binomial`)
- `-S`: Current spot price of the underlying asset
- `-K`: Strike price
- `-T`: Time to maturity (in years)
- `-r`: Risk-free interest rate (annualized)
- `--sigma`: Volatility (annualized)
- `-q`: Continuous dividend yield
- `-n`: Number of binomial tree steps
- `--backend`: Computation backend (`cpu`, `openmp`, or `cuda`)

### Running Benchmarks

```bash
# Run all benchmarks with "easy" parameters
./pricing_cli benchmark --parameters easy

# Filter benchmarks by function name
./pricing_cli benchmark --filter-by-name "american" --parameters medium
```

### Listing Available Benchmark Parameters

```bash
./pricing_cli parameters
```

## Architecture

### Pricing Models

The library currently implements the **Cox-Ross-Rubinstein (CRR) binomial model**, which is a discrete-time model for option pricing. The binomial tree method:

1. Divides time to maturity into `n` discrete steps
2. Models the underlying asset price as a recombining binomial tree
3. Calculates option values by backward induction from maturity to present
4. For American options, checks for early exercise at each node

### Backend Implementations

#### CPU Backend
Single-threaded implementation optimized for sequential processing. Suitable for small problem sizes or when parallelization overhead is not justified.

#### OpenMP Backend
Multi-threaded CPU implementation using OpenMP for parallel processing. Distributes computation across available CPU cores for improved performance on larger problems.

#### CUDA Backend
GPU-accelerated implementation using NVIDIA CUDA. Leverages massive parallelism for high-performance computing on large binomial trees.

## Development

### Install Pre-Commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit (using pip)
pip install pre-commit

# Or using Homebrew on macOS
brew install pre-commit

# Install the git hooks
pre-commit install
```

The hooks will automatically run:
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Large file checks
- clang-format for C++ code formatting

### Code Style

The project uses `clang-format` for consistent code formatting. Configuration is in `.clang-format`.

### Testing

Tests are written using the Catch2 framework:

```bash
# Run all tests
cd build
ctest

# Run specific test
./tests/test_vanilla_american_binomial
```

### Adding New Backends

To add a new backend:

1. Create backend implementation in `src/backends/<backend_name>/`
2. Add corresponding header in `include/backends/<backend_name>/`
3. Update `constants.hpp` to include the new backend enum
4. Link the backend in `src/models/CMakeLists.txt`
5. Add tests and benchmarks

## Mathematical Background

The Cox-Ross-Rubinstein binomial model uses these key formulas:

- **Up factor**: `u = exp(σ√Δt)`
- **Discount factor**: `disc = exp(-rΔt)`
- **Risk-neutral probabilities**: 
  - `p₀ = (u·exp(-qΔt) - disc) / (u² - 1)`
  - `p₁ = disc - p₀`

Where:
- `σ` is volatility
- `Δt = T/n` is the time step
- `r` is the risk-free rate
- `q` is the dividend yield

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style (enforced by clang-format)
2. All tests pass
3. New features include appropriate tests
4. Pre-commit hooks are installed and passing

## License

[License information to be added]

## References

- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A simplified approach." Journal of Financial Economics, 7(3), 229-263.
