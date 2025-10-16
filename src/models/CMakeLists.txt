
# Build a small models lib from src/models
file(GLOB MODELS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/models/*.cpp"
)

add_library(options_models STATIC ${MODELS_SOURCES})
target_include_directories(options_models
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
)
