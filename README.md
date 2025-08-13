# GPU_PIV
This is a C++ program exploiting GPUs to perform PIV on tiff images.

## Usage
The `GPU_PIV` program takes a single text file as an input. This text must contain the necessary settings for the PIV you wish to perform.
An example file can be found at `./example_resources/PIVsetup.in`. This file can be run by typing the following command into the command line from the root directory of the project

```
./build/bin/GPU_PIV ./example_resources/PIVsetup.in
```
The output of this can be found at: `./example_resources/vec_000.dat`

## Building the project
The program uses the following libraries that will need to be installed to build successfully
* OpenCL
* TIFF
* OpenMP
* fmt
To then build the project, run the INSTALL.sh bash script

