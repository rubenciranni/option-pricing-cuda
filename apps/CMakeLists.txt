# CLI-only for apps
include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(CLI11)

# App(s)
add_executable(binomial_crr_american_vanilla_option
  ${CMAKE_SOURCE_DIR}/apps/binomial_crr_american_vanilla_option.cpp
)

# Headers for the app
target_include_directories(binomial_crr_american_vanilla_option
  PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

# Link: always CPU; optionally OpenMP; and CLI11 only here
target_link_libraries(binomial_crr_american_vanilla_option
  PRIVATE
    CLI11::CLI11
)

# Link benchmarks first so that the cpu/openmp impl libraries (listed after)
# can resolve any symbols referenced by the benchmarks archive.
if (TARGET benchmarks_impl)
  target_link_libraries(binomial_crr_american_vanilla_option PRIVATE benchmarks_impl)
endif()

# Link runtime implementations after benchmarks so their symbols are available
target_link_libraries(binomial_crr_american_vanilla_option
  PRIVATE
    cpu_impl
)

if (TARGET openmp_impl)
  target_link_libraries(binomial_crr_american_vanilla_option PRIVATE openmp_impl)
endif()

set_target_properties(binomial_crr_american_vanilla_option PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
