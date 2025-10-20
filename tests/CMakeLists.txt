include(CTest)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(test_option_pricing ${TEST_SOURCES})

target_link_libraries(test_option_pricing
    PRIVATE
        models
)

target_include_directories(test_option_pricing
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_test(NAME test_option_pricing COMMAND test_option_pricing)
