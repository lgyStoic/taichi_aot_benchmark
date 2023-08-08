cmake_minimum_required(VERSION 3.13)

project(TaichiAotDemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

####################################################
# Configure Environment: Taichi C-API Installation #
####################################################
set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})
find_library(taichi_c_api taichi_c_api HINTS
    ${TAICHI_C_API_INSTALL_DIR}/lib
    NO_CMAKE_FIND_ROOT_PATH)

if (NOT EXISTS ${taichi_c_api})
    message(FATAL_ERROR "Couldn't find C-API library; ensure your Taichi is built with `TI_WITH_CAPI=ON`")
endif()

##########################
# Build for Android Demo #
##########################
set(TAICHI_AOT_DEMO_NAME "aotDemo")
set(PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")

list(APPEND TAICHI_AOT_DEMO_SRC
  ${PROJECT_ROOT_DIR}/pc/app.cpp
  ${PROJECT_ROOT_DIR}/pc/gemm.cpp
  ${PROJECT_ROOT_DIR}/pc/nlmean.cpp
  ${PROJECT_ROOT_DIR}/pc/reduce_sum.cpp
  ${PROJECT_ROOT_DIR}/pc/gaussian_filter.cpp
)
message(${PROJECT_ROOT_DIR}/OpenCV-android-sdk/sdk/native/jni)
set(OpenCV_DIR ${PROJECT_ROOT_DIR}/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_executable(${TAICHI_AOT_DEMO_NAME} ${TAICHI_AOT_DEMO_SRC})
target_include_directories(${TAICHI_AOT_DEMO_NAME} PUBLIC
  ${TAICHI_C_API_INSTALL_DIR}/include)

target_link_libraries(${TAICHI_AOT_DEMO_NAME} ${taichi_c_api} ${OpenCV_LIBS})

set_target_properties(${TAICHI_AOT_DEMO_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

if (ANDROID)
  target_link_libraries(${TAICHI_AOT_DEMO_NAME} android log)
endif()
