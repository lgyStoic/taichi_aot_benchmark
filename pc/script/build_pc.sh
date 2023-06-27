#!/bin/bash
set -e

rm -rf build
mkdir build
pushd build

TAICHI_PROJECT_DIR="${PWD}/.." cmake -DCMAKE_BUILD_TYPE=Release .. 
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
