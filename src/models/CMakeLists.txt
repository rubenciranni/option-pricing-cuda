file(GLOB MODELS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(models STATIC)

target_sources(models PRIVATE ${MODELS_SOURCES})

target_include_directories(models
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(models
    PUBLIC
      cpu_impl
      openmp_impl
      # cuda_impl
)
<<<<<<< Updated upstream
=======

target_compile_features(openmp_impl PUBLIC cxx_std_17)
>>>>>>> Stashed changes
