
# OpenMP backend implementation (guarded at top level)
find_package(OpenMP REQUIRED)

add_library(openmp_impl STATIC
  ${CMAKE_SOURCE_DIR}/src/backends/openmp/binomial_crr_american_vanilla_option_openmp.cpp
)

target_include_directories(openmp_impl
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(openmp_impl
  PUBLIC
    options_models
    OpenMP::OpenMP_CXX
)

# If you also want to propagate -fopenmp to dependents explicitly:
target_compile_features(openmp_impl PUBLIC cxx_std_17)
