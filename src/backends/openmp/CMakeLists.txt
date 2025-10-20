find_package(OpenMP REQUIRED)

file(GLOB_RECURSE OPENMP_BACKEND_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(openmp_impl STATIC)

target_sources(openmp_impl PRIVATE ${OPENMP_BACKEND_SOURCES})

target_include_directories(openmp_impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(openmp_impl
  PUBLIC
    OpenMP::OpenMP_CXX
)

target_compile_features(openmp_impl PUBLIC cxx_std_17)
