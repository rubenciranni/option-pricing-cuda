include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.5.0
)
# Declare rang (header-only)
FetchContent_Declare(
    rang
    GIT_REPOSITORY https://github.com/agauniyal/rang.git
    GIT_TAG        master   # you can pin a commit hash instead for reproducibility
)

FetchContent_MakeAvailable(rang)

FetchContent_MakeAvailable(CLI11)

add_executable(pricing_cli)

target_sources(pricing_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pricing_cli.cpp
)

target_include_directories(pricing_cli
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pricing_cli
    PRIVATE
      CLI11::CLI11
      rang
      benchmarks
      models
      cpu_impl
)

if(TARGET openmp_impl)
    target_link_libraries(pricing_cli PRIVATE openmp_impl)
endif()

if(TARGET cuda_impl)
    target_link_libraries(pricing_cli PRIVATE cuda_impl)
endif()

set_target_properties(pricing_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

target_compile_features(pricing_cli PUBLIC cxx_std_17)
