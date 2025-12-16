# add_compile_definitions(_GLIBCXX_USE_FLOAT_128=0)
set(CMAKE_CUDA_ARCHITECTURES 120)
find_package(CUDAToolkit REQUIRED)
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

# Declare modular header dependencies for better caching and rebuild tracking
set_source_files_properties(${CUDA_BACKEND_SOURCES}
    PROPERTIES
    OBJECT_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/backends/cuda/vanilla_american_binomial_cuda.cuh;${PROJECT_SOURCE_DIR}/include/backends/cuda/vanilla_american_binomial_cuda_basic.cuh;${PROJECT_SOURCE_DIR}/include/backends/cuda/vanilla_american_binomial_cuda_templated.cuh;${PROJECT_SOURCE_DIR}/include/backends/cuda/vanilla_american_binomial_cuda_batch.cuh"
)

target_compile_options(cuda_impl PRIVATE
    -Wno-pedantic
)

set_target_properties(cuda_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_features(cuda_impl PUBLIC cxx_std_17)
target_compile_options(cuda_impl PRIVATE
    --ptxas-options=-v
    --use_fast_math
    -lnvToolsExt
)
target_link_libraries(cuda_impl PUBLIC CUDA::cudart CUDA::cuda_driver)
