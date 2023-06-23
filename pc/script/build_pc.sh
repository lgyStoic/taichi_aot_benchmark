#!/bin/bash
set -e

rm -rf build
mkdir build
pushd build
TAICHI_C_API_INSTALL_DIR="${PWD}/../c_api" cmake ..
if [ $? -ne 0 ]; then
    echo "Configuration failed"
    exit -1
fi

cmake --build .
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit -1
fi
popd
