# shell.nix
{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

pkgs.mkShell {
  name = "cpp-omp-cmake";
  packages = with pkgs; [
    gcc          # includes g++
    gdb
    cmake
    pkg-config
    cudaPackages.cuda_nvvp
    cudaPackages.cuda_cudart
    # glibc is provided by gcc's stdenv on Linux; no need to add explicitly
    clang-tools   # adds clangd, clang-tidy, etc.
  ];

  shellHook = ''
    export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH
    export CUDA_HOME=${pkgs.cudaPackages.cuda_cudart}
    export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH

    export LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LIBRARY_PATH
  '';

  # Make sure CMake picks GCC by default
  CC = "gcc";
  CXX = "g++";
}
