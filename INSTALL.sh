#!/bin/bash

# Check if the build directory exists
if [ -d "build" ]; then
    echo "Deleting existing build directory."
    # Remove the build directory and its contents
    rm -rf build
fi

# Create a new, clean build directory
mkdir build
cd build

echo "Configuring the project with CMake..."
cmake ..

echo "Building the project..."
cmake --build . --config Release

echo "Build process complete."
