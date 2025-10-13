cmake_minimum_required(VERSION 3.22)
project(QuantOptions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

enable_testing()

add_subdirectory(src/cpu)
add_subdirectory(tests)
# add_subdirectory(src/cuda)

add_executable(binomial_crr_american_vanilla_option
    src/binomial_crr_american_vanilla_option.cpp
)

target_link_libraries(binomial_crr_american_vanilla_option
    cpu_impl
)

set_target_properties(binomial_crr_american_vanilla_option PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
