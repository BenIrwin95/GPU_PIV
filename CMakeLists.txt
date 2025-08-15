# Minimum CMake version required
cmake_minimum_required(VERSION 3.15)

# Define the project name and languages
project(GPU_PIV_Project LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Use strict C++20

# Add common compiler flags
add_compile_options(-Wall -g)

# Source files
set(SRCS_main
    src/main.cpp
    src/OpenCL_utilities.cpp
    src/inputFunctions.cpp
    src/tiffFunctions.cpp
    src/dataArrangement.cpp
    src/bicubic_interpolation.cpp
    src/FFT.cpp
    src/complexMaths.cpp
    src/determineCorrelation.cpp
    src/vectorValidation.cpp
    src/outputFunctions.cpp
)

set(SRCS_im_filter
    src/main_imagePreproces.cpp
    src/OpenCL_utilities.cpp
    src/inputFunctions.cpp
    src/tiffFunctions.cpp
    src/dataArrangement.cpp
    src/bicubic_interpolation.cpp
    src/FFT.cpp
    src/complexMaths.cpp
    src/determineCorrelation.cpp
    src/vectorValidation.cpp
    src/outputFunctions.cpp
)

# Define the executable target
add_executable(GPU_PIV ${SRCS_main})
add_executable(IM_FILTER ${SRCS_im_filter})

# --- Find and link libraries ---

# Find TIFF library
find_package(TIFF REQUIRED)
if (TIFF_FOUND)
    target_link_libraries(GPU_PIV PRIVATE ${TIFF_LIBRARIES})
    target_include_directories(GPU_PIV PRIVATE ${TIFF_INCLUDE_DIR})
    target_link_libraries(IM_FILTER PRIVATE ${TIFF_LIBRARIES})
    target_include_directories(IM_FILTER PRIVATE ${TIFF_INCLUDE_DIR})
    message(STATUS "Found TIFF: ${TIFF_LIBRARIES}")
else()
    message(FATAL_ERROR "TIFF library not found. Please install libtiff development files.")
endif()

# Find OpenCL library
find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
    target_link_libraries(GPU_PIV PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(GPU_PIV PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(IM_FILTER PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(IM_FILTER PRIVATE ${OpenCL_INCLUDE_DIRS})
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenCL library not found. Please install OpenCL SDK.")
endif()

# Find fmt library
find_package(fmt CONFIG REQUIRED) # Use CONFIG mode for modern fmt library
if (fmt_FOUND)
    target_link_libraries(GPU_PIV PRIVATE fmt::fmt)
    target_link_libraries(IM_FILTER PRIVATE fmt::fmt)
    message(STATUS "Found fmt: fmt::fmt")
else()
    message(FATAL_ERROR "fmt library not found. Please install fmt development files.")
endif()

# Link against the math library (often implicit on Linux, but good to be explicit for cross-platform)
# On Windows, this is often part of the compiler's runtime, so it might not be explicitly needed.
if (NOT WIN32)
    target_link_libraries(GPU_PIV PRIVATE m)
    target_link_libraries(IM_FILTER PRIVATE m)
endif()


# --- Find and link HDF5 library ---
# The COMPONENTS keyword tells CMake to look for both the C and C++ bindings.
find_package(HDF5 REQUIRED COMPONENTS CXX)
if (HDF5_FOUND)
    # Link the executable against the HDF5 libraries.
    # The HDF5_LIBRARIES variable includes both hdf5_cpp and hdf5.
    target_link_libraries(GPU_PIV PRIVATE ${HDF5_LIBRARIES})
    # Add the include directories. The HDF5_INCLUDE_DIRS variable contains the paths for both C and C++ headers.
    target_include_directories(GPU_PIV PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(IM_FILTER PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(IM_FILTER PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "Found HDF5: ${HDF5_LIBRARIES}")
else()
    message(FATAL_ERROR "HDF5 library not found. Please install hdf5-devel.")
endif()


# --- Handle OpenMP ---
find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(GPU_PIV PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(GPU_PIV PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(IM_FILTER PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(IM_FILTER PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP support enabled.")
else()
    message(WARNING "OpenMP not found or not supported by compiler. Program will compile without OpenMP acceleration.")
endif()

# --- Set output directory and name for the executable ---
# CMake automatically adds .exe extension on Windows
set_target_properties(GPU_PIV PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
set_target_properties(IM_FILTER PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Optional: Install target
install(TARGETS GPU_PIV DESTINATION bin)
install(TARGETS IM_FILTER DESTINATION bin)
