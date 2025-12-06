
{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:
pkgs.mkShell {
  name = "cpp-omp-cmake";
  packages = with pkgs; [
    gcc          # includes g++
    gdb
    cmake
    ninja
    pkg-config
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_nvvp
    cudaPackages.cuda_cudart
    cudaPackages.cuda_profiler_api  # Add this for cuda_profiler_api.h
    stdenv.cc.cc.lib
    uv
    cudaPackages.cuda_nvtx          # For nvToolsExt.h
    clang-tools   # adds clangd, clang-tidy, etc.
  ];
  shellHook = ''
    export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH
    export CUDA_HOME=${pkgs.cudaPackages.cuda_cudart}
    export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LIBRARY_PATH
      export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    # Add include paths for profiler and NVTX headers
    export CPATH=${pkgs.cudaPackages.cuda_profiler_api}/include:${pkgs.cudaPackages.cuda_nvtx}/include:$CPATH
  '';
  # Make sure CMake picks GCC by default
  CC = "gcc";
  CXX = "g++";
}
