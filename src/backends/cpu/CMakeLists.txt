
# CPU backend implementation
add_library(cpu_impl STATIC
  ${CMAKE_SOURCE_DIR}/src/backends/cpu/binomial_crr_american_vanilla_option_cpu.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/cpu/binomial_crr_european_vanilla_option_cpu.cpp
)

target_include_directories(cpu_impl
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(cpu_impl
  PUBLIC
    options_models           # depend on models
)
