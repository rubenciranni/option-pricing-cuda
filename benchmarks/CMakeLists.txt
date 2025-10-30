file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(benchmarks STATIC)

target_sources(benchmarks PRIVATE ${BENCHMARK_SOURCES})

target_include_directories(benchmarks
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
