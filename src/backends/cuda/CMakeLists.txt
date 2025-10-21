# add_compile_definitions(_GLIBCXX_USE_FLOAT_128=0)
set(CMAKE_CUDA_ARCHITECTURES 90)
enable_language(CUDA)

file(GLOB_RECURSE CUDA_BACKEND_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_library(cuda_impl STATIC)

target_sources(cuda_impl PRIVATE ${CUDA_BACKEND_SOURCES})

target_include_directories(cuda_impl
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(cuda_impl PRIVATE
    -Wno-pedantic
)

set_target_properties(cuda_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_features(cuda_impl PUBLIC cxx_std_17)
