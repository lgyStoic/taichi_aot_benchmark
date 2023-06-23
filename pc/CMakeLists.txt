cmake_minimum_required(VERSION 3.17)

set(TAICHI_AOT_APP_NAME TaichiAot)
project(${TAICHI_AOT_APP_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})
find_library(taichi_c_api taichi_c_api HINTS
    ${TAICHI_C_API_INSTALL_DIR}/lib
    NO_CMAKE_FIND_ROOT_PATH)

message("taichi c api path: ${TAICHI_C_API_INSTALL_DIR}")
if (NOT EXISTS ${taichi_c_api})
    message(FATAL_ERROR "Couldn't find C-API library; ensure your Taichi is built with `TI_WITH_CAPI=ON`")
endif()


# Declare executable target.
add_executable(${TAICHI_AOT_APP_NAME} app.cpp)
target_include_directories(${TAICHI_AOT_APP_NAME} PUBLIC ${TAICHI_C_API_INSTALL_DIR}/include)
target_link_libraries(${TAICHI_AOT_APP_NAME} ${taichi_c_api})

set_target_properties(${TAICHI_NERF_DEMO_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
