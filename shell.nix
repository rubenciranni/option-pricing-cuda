# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "cpp-omp-cmake";
  packages = with pkgs; [
    gcc          # includes g++
    gdb
    cmake
    pkg-config
    # glibc is provided by gcc's stdenv on Linux; no need to add explicitly
  ];

  # Make sure CMake picks GCC by default
  CC = "gcc";
  CXX = "g++";
}

