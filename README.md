# Option Pricing CUDA


## Compilation Instructions

> For nix-os based systems (Linux, macOS) run `nix-shell` to enter a development shell with all dependencies installed. 😉

```bash
mkdir build
cd build
cmake ..
make
```

## Usage Instructions


## Single Option Pricing


 So there are three commands:
- `single`: for pricing a single option you can specify parameters the function and the function name (**manly for testing debugging**)
- `benchmark`: for running benchmarks on a dataset
- `list`: for listing available datasets