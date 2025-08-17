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
std::vector<float> findFloatsAfterKeyword(const std::string& filename, const std::string& keyword);
std::vector<std::string> separate_words(std::string inputString);

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
void writeTiffFromAppropriateIntegerVector(const ImageData& imageData, const std::string& filePath);
cl_int retrieveImageFromBuffer(cl::Buffer& im_buffer_complex, cl::Buffer& im_buffer, ImageData& im, OpenCL_env& env);


//--------------------------------------------------------------------------------
//-------------------------------dataArrangement----------------------------------
//--------------------------------------------------------------------------------

cl_int uniformly_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env);
cl_int warped_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env);

cl_int detrend_windows(cl::Buffer& input, cl_int2 inputDim, int windowSize, cl_int2 arrSize, OpenCL_env& env);
//--------------------------------------------------------------------------------
//--------------------------bicubic_interpolation---------------------------------
//--------------------------------------------------------------------------------

cl_int determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, uint32_t im_width, uint32_t im_height);
cl_int upscale_velocity_field(int pass, PIVdata& piv_data, OpenCL_env& env);

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
void add_pass_data_to_file(int pass, std::ofstream& outputFile, PIVdata& piv_data);
H5::H5File initialise_output_hdf5(const std::string filename, PIVdata& piv_data);
void add_data_to_hdf5_output(int frame, int pass, H5::H5File& file, PIVdata& piv_data);


//--------------------------------------------------------------------------------
//-------------------------------image_processing---------------------------------
//--------------------------------------------------------------------------------

ImFilter createFilter(std::vector<std::string>& words, OpenCL_env& env);
std::vector<ImFilter> create_filter_list(int N_filter, std::string inputFile, OpenCL_env& env);
cl_int process_image_with_filterList(cl::Buffer& im_buffer_complex, ImageData& im, std::vector<ImFilter>& filter_list, OpenCL_env& env);
cl_int runFilter(cl::Buffer& im_buffer_complex, ImageData& im, ImFilter& filter, OpenCL_env& env);

cl_int manual_range_scaling(cl::Buffer& im_buffer, ImageData& im, float minVal, float maxVal, OpenCL_env& env);
cl_int image_mean_filter(cl::Buffer& im_buffer_complex, ImageData& im, int radius, OpenCL_env& env);
cl_int image_mean_filter_subtraction(cl::Buffer& im_buffer_complex, ImageData& im, int radius, OpenCL_env& env);
cl_int image_gauss_filter(cl::Buffer& im_buffer_complex, ImageData& im, int radius, float stdDev, OpenCL_env& env);

#endif
