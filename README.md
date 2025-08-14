# GPU_PIV
This is a C++ program exploiting GPUs to perform PIV on tiff images.

## Usage
The `GPU_PIV` program takes a single text file as an input. This text must contain the necessary settings for the PIV you wish to perform.
An example file can be found at `./example_resources/PIVsetup.in`. This file can be run by typing the following command into the command line from the root directory of the project

```
./build/bin/GPU_PIV ./example_resources/PIVsetup.in
```
The output of this can be found at: `./example_resources/vec_000.dat`. Note that on Windows you will need to replace `/` with `\`

## Building the project
The program uses the following libraries that will need to be installed to build successfully
* OpenCL
* TIFF
* OpenMP
* fmt
* HDF5

To then build the project,if on Linux, you can run the INSTALL.sh bash script

### Building on Windows
For installing the necessary libraries on windows I would recommend using vcpkg.
Installing vcpkg can be done with:
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```
To install the libraries, then run:
```
vcpkg install tiff
vcpkg install opencl
vcpkg install fmt
vcpkg install hdf5:x64-windows
```
To then build GPU_PIV, change to the project's root folder, make sure the build folder has been removed, and then run:
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cd build 
cmake --build . --config Release
```

### Building on Mac (untested)
Installing the dependencies can be done through homebrew:
```
brew install tiff fmt hdf5 opencl-clhpp-headers
```

To then build GPU_PIV, change to the project's root folder, make sure the build folder has been removed, and then run:
```
cmake -B build -S .
cd build
cmake --build . --config Release
```

