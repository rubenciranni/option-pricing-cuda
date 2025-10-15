

cmake_minimum_required(VERSION 3.22)
project(QuantOptions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(CLI11)

include_directories(${CMAKE_SOURCE_DIR}/include)

enable_testing()

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
add_subdirectory(src/cpu)

file(GLOB SRC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(binomial_crr_american_vanilla_option ${SRC_SOURCES})

target_link_libraries(binomial_crr_american_vanilla_option
    PRIVATE
        cpu_impl
        CLI11::CLI11
)

set_target_properties(binomial_crr_american_vanilla_option PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
