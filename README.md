# GPU_PIV
This is a C++ program exploiting GPUs to perform PIV on tiff images.

## Quickstart
The `GPU_PIV` program takes a single text file as an input. This text must contain the necessary settings for the PIV you wish to perform.
An example file can be found at `./example_resources/PIVsetup.in`. This file can be run by typing the following command into the command line from the root directory of the project

```
./build/bin/GPU_PIV ./example_resources/PIVsetup.in
```
The output of this can be found at: `./example_resources/output.h5`. 

Note that on Windows you will need to replace `/` with `\` and use the `.exe` suffix: `GPU_PIV.exe`.


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



## Usage: Creating an input file

GPU_PIV works by taking a text-file as an input that describes what images need to be processed, with what settings they should be processed and finally where the output of the program should be saved. An example file `PIVsetup.in` can be found in the example_resources folder. These next sections will go over the different sections this file needs and what options there are. All inputs consist of a keyword defining what variable is being set followed the value it is being set too. There can only be ONE keyword per line.

### Debug Level

```
DEBUG 2
```

GPU_PIV will print status messages as it runs to describe the different tasks it is performing as well as timing how long certain sections take to run. The debug level must be a single integer. By increasing its value, increasingly more status messages will be shown

### Image settings

```
N_FRAMES 1

IMAGEFILE_1 ./example_resources/cam1_im_{:03}_A.tiff
IM1_FRAME_START 0
IM1_FRAME_STEP 1

IMAGEFILE_2 ./example_resources/cam1_im_{:03}_B.tiff
IM2_FRAME_START 0
IM2_FRAME_STEP 1
```

As GPU_PIV runs, it fetches images via formatting `IMAGEFILE_1` and `IMAGEFILE_2` with the `fmt::format()` function. The number inserted into these strings is calculated from the frame index:
```
for(int frame=0;frame<N_frames;frame++){
    int frame_index_for_image_1 = IM1_FRAME_START + frame * IM1_FRAME_STEP;
    int frame_index_for_image_2 = IM2_FRAME_START + frame * IM2_FRAME_STEP;
    ...
    
}
```

In the example input file `{:03}` will format the frame index into a 3 digit integer, replacing empty digits with zeros.


### PIV settings
```
N_PASS 2
WINDOW_SIZE 32 16
WINDOW_OVERLAP 0.5 0.5
```

GPU_PIV uses a multi-pass method to account for particle displacements greater than the window size. 

`N_PASS` must be a single integer and represents how many sequential passes will be performed.

`WINDOW_SIZE` sets the side length of the interrogation windows. Must be an integer power of 2

`WINDOW_OVERLAP` sets how much neighbouring interrogation windows may overlap. Must be a float from 0 < WINDOW_OVERLAP < 1, where 0.5 represents 50% overlap

### Output settings
```
OUTPUT_TEMPLATE ./example_resources/vec_{:03}.dat
SAVE_FORMAT ASCII
```
Or
```
OUTPUT_TEMPLATE ./example_resources/output.h5
SAVE_FORMAT HDF5
```

GPU_PIV has two ways of saving its output. The first method as a text-file output that is saved for every frame pair. The format of the ouput is readable and self-explanatory but prewritten code for interpreting them using either Matlab or Python can be found in the example_resources folder.

The second method saves a single HDF5 file that contains the outputs of all frame-pairs and passes. The hierachical structure of HDF5 allows all of this data to be stored in a organised manner and extracted piecemeal as necessary. Examples for how to extract the data can also be found in the example_resources folder.

The HDF5 method is notably faster than the ASCII method, but the ASCII format has been left for those for who prefer it.


## Work in Progress and known bugs
* Image pre-processing
* Something doesn't seem to work quite right for uint16 tiff images
