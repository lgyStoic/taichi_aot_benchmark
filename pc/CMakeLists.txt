cmake_minimum_required(VERSION 3.17)

set(TAICHI_AOT_APP_NAME TaichiAot)
project(${TAICHI_AOT_APP_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (APPLE AND (NOT IOS))
    set(OSX ON)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
        if (CMAKE_CUDA_COMPILER)
            set(TEGRA ON)
        else()
            set(QUALCOMM ON)
        endif()
    else ()
        set(LINUX ON)
    endif ()
endif ()

set(TAICHI_PROJECT_DIR $ENV{TAICHI_PROJECT_DIR})
if (LINUX)
	set(TAICHI_C_API_INSTALL_DIR ${TAICHI_PROJECT_DIR}/c_api/linux/)
	find_library(taichi_c_api taichi_c_api HINTS
		${TAICHI_C_API_INSTALL_DIR}/lib
		NO_CMAKE_FIND_ROOT_PATH)

elseif (OSX)
	set(TAICHI_C_API_INSTALL_DIR ${TAICHI_PROJECT_DIR}/c_api/osx/)
	find_library(taichi_c_api taichi_c_api HINTS
		${TAICHI_C_API_INSTALL_DIR}/lib
		NO_CMAKE_FIND_ROOT_PATH)
endif()


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


# set assets
set(asset_source "${CMAKE_SOURCE_DIR}/../taichi_algorithm/")
set(asset_dest "${CMAKE_CURRENT_BINARY_DIR}/assets")
file(COPY ${asset_source} DESTINATION ${asset_dest})
