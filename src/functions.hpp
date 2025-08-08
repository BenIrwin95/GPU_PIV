#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include "standardLibraries.hpp"
#include "globalVars.hpp"

//--------------------------------------------------------------------------------
//-------------------------------inputFunctions---------------------------------
//--------------------------------------------------------------------------------

int findIntegerAfterKeyword(const std::string& filename, const std::string& keyword);
std::string findRestOfLineAfterKeyword(const std::string& filename, const std::string& keyword);
std::vector<int> findIntegersAfterKeyword(const std::string& filename, const std::string& keyword);

//--------------------------------------------------------------------------------
//-------------------------------OpenCL_utilities---------------------------------
//--------------------------------------------------------------------------------
const char* get_cl_error_string(cl_int err);
void print_cl_error(cl_int err, const std::string& filename, int line_number);
cl_int inititialise_OpenCL_buffers(OpenCL_env& env, PIVdata& piv_data, ImageData& im);


//--------------------------------------------------------------------------------
//-------------------------------tiffFunctions------------------------------------
//--------------------------------------------------------------------------------

ImageData readTiffToAppropriateIntegerVector(const std::string& filePath);
cl_int uploadImage_and_convert_to_complex(ImageData& im, OpenCL_env& env, cl::Buffer& buffer, cl::Buffer& buffer_complex);


//--------------------------------------------------------------------------------
//-------------------------------dataArrangement----------------------------------
//--------------------------------------------------------------------------------

cl_int uniformly_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env);

#endif
