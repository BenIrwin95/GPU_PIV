#!/bin/bash

# Check if the build directory exists
if [ -d "build" ]; then
    echo "Deleting existing build directory."
    # Remove the build directory and its contents
    rm -rf build
fi

# Create a new, clean build directory
echo "Configuring the project with CMake"
cmake -B build -S .
cd build

echo "Building the project"
cmake --build . --config Release
cd ..
echo "Build process complete."
