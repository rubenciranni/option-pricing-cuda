file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(benchmarks_impl STATIC)

target_sources(benchmarks_impl PRIVATE ${BENCHMARK_SOURCES})

target_include_directories(benchmarks_impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
