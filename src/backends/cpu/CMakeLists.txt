file(GLOB_RECURSE CPU_BACKEND_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(cpu_impl STATIC)

target_sources(cpu_impl PRIVATE ${CPU_BACKEND_SOURCES})

target_include_directories(cpu_impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cpu_impl PUBLIC cxx_std_17)
