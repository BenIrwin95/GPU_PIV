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



# Table of Contents
* [Quickstart](#quickstart)
* [Building the project](#building-the-project)
    * [Building on Windows](#building-on-windows)
    * [Building on Mac (untested)](#building-on-mac-untested)
* [Usage: Creating an input file](#usage-creating-an-input-file)
    * [Debug Level](#debug-level)
    * [Image settings](#image-settings)
    * [PIV settings](#piv-settings)
    * [Output settings](#output-settings)
* [Image Pre-processor Usage](#image-pre-processor-usage)
    * [Image loading/saving](#image-loadingsaving)
    * [Image statistics](#image-statistics)
    * [Setting number of filters](#setting-number-of-filters)
    * [Manual range scaling](#manual-range-scaling)
    * [Mean filtering](#mean-filtering)
    * [Gaussian filtering](#gaussian-filtering)
* [Work in Progress and known bugs](#work-in-progress-and-known-bugs)




## Building the project
The program uses the following libraries that will need to be installed to build successfully
* OpenCL
* TIFF
* OpenMP
* fmt
* HDF5

To then build the project, if on Linux, you can run the INSTALL.sh bash script

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

Note that any relative filepaths (ie. using `./`) must be given relative to where GPU_PIV is run from NOT where PIVsetup.in is stored

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



## Image Pre-processor Usage

GPU_PIV is able to apply a select few types of filters to the images loaded into it. These filters can be tested in standalone upon some test image through the IM_FILTER executable. Much like GPU_PIV, it takes a textfile as an input with specifies what image it loads and how it filters. With the exception of `IMAGE_SRC` and `IMAGE_DST`, all commands relating to the filter setup can also be inputted into the GPU_PIV input file. GPU_PIV will only use these filters however if `ACTIVATE_IM_FILTER` is set to 1 in the inputfile.

The following sections will go through the input commands that the IM_FILTER inputfile can take and what they mean. An example file can be found at `./example_resources/IM_FILTER_setup.in`.


### Image loading/saving
```
IMAGE_SRC ./example_resources/RecordedImage_GO-5100M-USB__000.tif
IMAGE_DST ./example_resources/test.tiff
```
`IMAGE_SRC` and `IMAGE_DST` set the image that should be loaded and under what name it should be saved respectively.


### Image statistics
```
DISPLAY_IMAGE_STATISTICS 0
```

For setting the arguments to some filters it can be useful to know some statistics about the pixel intensities in an image. By setting `DISPLAY_IMAGE_STATISTICS` to 1, these statistics will be calculated and displayed. If not required, set to zero.

### Setting number of filters
```
N_FILTER 2
FILTER_0 ...
```

To allow for multiple filters to be applied to an image in a particular order, the number of applied filters must be specified. The program will then iterate starting from 0 looking for lines that start with `FILTER_{}`, which then set the type of filter being applied and its arguments.


### Manual range scaling
```
//FILTER_0 MANUAL_STRETCH min max
FILTER_0 MANUAL_STRETCH 0.0 0.1
```

Rescales the pixel intensities between the minimum and maximum values for the image's datatype based on two float thresholds: min and max. Everything equal or below min will be set to zero, while everything equal and above max will be set the image datatype's maximum value. All inbetween values will be linearly interpolated.

min and max should be floats in the range (0,1). In the example input shown above, all values above 10% of the datatype's maximum value will be scaled up to the datatype's maximum value.


### Mean filtering
```
//FILTER_0 MEAN_FILTER radius
FILTER_0 MEAN_FILTER 3
//FILTER_1 MEAN_FILTER_SUBTRACTION radius
FILTER_1 MEAN_FILTER_SUBTRACTION 3
```

Using `MEAN_FILTER` the intensity of every pixel will be set equal to the mean of a (2*radius+1, 2*radius+1) square window centered on the pixel. Conversely, `MEAN_FILTER_SUBTRACTION` will subtract this mean filtered value from the original pixel intensity.


### Gaussian filtering
```
//FILTER_0 GAUSS_FILTER radius stdDev
FILTER_0 GAUSS_FILTER 3 1
//FILTER_1 GAUSS_FILTER_SUBTRACTION radius stdDev
FILTER_1 GAUSS_FILTER_SUBTRACTION 3 1
```

Very similar to mean filtering, but the pixel intensities within the window are weighted by a gaussian distribution with a standard deviation stdDev.

## Work in Progress and known bugs
* Image masking
