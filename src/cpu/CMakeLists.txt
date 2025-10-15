
file(GLOB CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)


add_library(cpu_impl ${CPU_SOURCES})

target_include_directories(cpu_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(cpu_impl PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS " cpu_impl built with OpenMP support")
else()
    message(STATUS " cpu_impl built without OpenMP")
endif()
