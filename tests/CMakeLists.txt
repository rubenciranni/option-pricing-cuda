include_directories(${PROJECT_SOURCE_DIR}/include)

set(TEST_SOURCES
    test_binomial_crr_american_vanilla_option_cpu.cpp
    test_binomial_crr_european_vanilla_option_cpu.cpp
)

add_executable(test_cpu ${TEST_SOURCES})
target_link_libraries(test_cpu PRIVATE cpu_impl)

add_test(NAME test_cpu COMMAND test_cpu)
