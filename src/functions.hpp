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
cl_int warped_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env);
//--------------------------------------------------------------------------------
//--------------------------Bspline_interpolation---------------------------------
//--------------------------------------------------------------------------------


alglib::spline2dinterpolant create_interpolater(std::vector<double>& X, std::vector<double>& Y, std::vector<double>& C);
splineInterp create2DSplineInterp(std::vector<double>& x, std::vector<double>& y, std::vector<double>& C);
double interpolate_2Dspline(double x, double y, splineInterp& spline);
void determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, std::vector<cl_int2>& imageShifts, uint32_t im_width, uint32_t im_height);

//--------------------------------------------------------------------------------
//------------------------------------FFT-----------------------------------------
//--------------------------------------------------------------------------------

cl_int FFT2D_tiled(cl::Buffer& input, cl_int2 inputDim, int windowSize, int dir, OpenCL_env& env);

//--------------------------------------------------------------------------------
//----------------------------determineCorrelation--------------------------------
//--------------------------------------------------------------------------------

cl_int FFT_corr_tiled(cl::Buffer& input1, cl::Buffer& input2, cl_int2 inputDim, int windowSize, OpenCL_env& env);
cl_int find_max_corr(cl::Buffer& input, cl_int2 inputDim, int windowSize, cl::Buffer& outputU, cl::Buffer& outputV, cl_int2 arrSize, int activate_subpixel, OpenCL_env& env);


//--------------------------------------------------------------------------------
//-------------------------------vectorValidation---------------------------------
//--------------------------------------------------------------------------------

cl_int validateVectors(int pass, PIVdata& piv_data, OpenCL_env& env);


//--------------------------------------------------------------------------------
//-------------------------------outputFunctions----------------------------------
//--------------------------------------------------------------------------------
void add_pass_data_to_file(int pass, std::ofstream& outputFile, PIVdata& piv_data, OpenCL_env& env);

#endif
