set(CPU_SOURCES
    binomial_crr_american_vanilla_option_cpu.cpp
    binomial_crr_european_vanilla_option_cpu.cpp
)

add_library(cpu_impl ${CPU_SOURCES})

target_include_directories(cpu_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
)
