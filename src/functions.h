#ifndef FUNCTIONS_H
#define FUNCTIONS_H


#include "standardLibraries.h"


//--------------------------------------------------------------------------------------------------
//--------------------------------------OpenCL_utilities--------------------------------------------
//--------------------------------------------------------------------------------------------------

const char* getOpenCLErrorString(cl_int err);

cl_int initialise_OpenCL(cl_platform_id *platform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue, cl_command_queue *queueNonBlocking, cl_program *program,cl_kernel *kernelFFT_1D, cl_kernel *kernelMultConj, cl_kernel *kernelMaxCorr,cl_kernel *kernel_uniformTiling, cl_kernel *kernel_offsetTiling,cl_kernel *kernel_identifyInvalidVectors, cl_kernel *kernel_correctInvalidVectors);

//--------------------------------------------------------------------------------------------------
//--------------------------------------utilities---------------------------------------------------
//--------------------------------------------------------------------------------------------------

// utility function to display a debug message if a debug variable is higher than some value
void debug_message(const char* message, int debug, int min_debug, clock_t* lastTime);

void round_float_array(float* A, int N);


void printComplexArray(cl_float2* input, size_t width, size_t height);


// function to multiply every element of a double array by a scalar value
void multiply_double_array_by_scalar(double* arr, int arrSize, double scalar);
void multiply_float_array_by_scalar(float* arr, int arrSize, float scalar);


//--------------------------------------------------------------------------------------------------
//--------------------------------------inputFunctions----------------------------------------------
//--------------------------------------------------------------------------------------------------

// function to remove newline characters from a string
void remove_newline(char *str);

// removes all whitespace from a string
void strip_whitespace_inplace(char *str);


// function to return the what follows on the line starting with a keyword
char* find_line_after_keyword(const char* filename, const char* keyword, int* status);

// function to return the integer directly following a specified keyword
int extract_int_by_keyword(const char* filename, const char* keyword, int* status);

// extract a known number of integers from a char array
// returns NULL if the incorrect number of integers is found
int* extract_integer_list_from_char(const char *str, int expected_count, int* status);

// extract a known number of integers from an input file based on a keyword the line starts with
// returns NULL if the incorrect number of integers is found
int* find_int_list_after_keyword(const char* filename, const char* keyword, const int N, int* status);


//--------------------------------------------------------------------------------------------------
//--------------------------------------tiffFunctions-----------------------------------------------
//--------------------------------------------------------------------------------------------------



/*readSingleChannelTiff
 * function to read a single-channel (grayscale) tiff file at a specified location and pass its contents into an array
 * first output points towards the bottom left of the image
 * the last output points towards the top right of the image
 */
uint8_t* readSingleChannelTiff(const char* filePath, uint32_t* width, uint32_t* height);



// function to convert a uint8_t image into complex form
// this is needed for FFT
cl_float2* tiff2complex(const uint8_t* input ,uint32_t width, uint32_t height);


int get_tiff_dimensions_single_channel(const char *filepath, uint32_t *width, uint32_t *height);


//--------------------------------------------------------------------------------------------------
//--------------------------------------dataArrangement---------------------------------------------
//--------------------------------------------------------------------------------------------------


void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int windowShift, cl_int2 vecDim, cl_mem output, cl_kernel kernel_uniformTiling, cl_command_queue queue);

void offset_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_mem U_GPU, cl_mem V_GPU, cl_int2 vecDim, cl_mem output, cl_kernel kernel_offsetTiling, cl_command_queue queue);


//--------------------------------------------------------------------------------------------------
//--------------------------------------determineCorrelation----------------------------------------
//--------------------------------------------------------------------------------------------------


void FFT_corr_tiled (cl_mem input1,cl_mem input2, cl_int2 inputDim, int windowSize, cl_kernel kernelFFT_1D, cl_kernel kernelMultiplyConj, cl_command_queue queue);


//--------------------------------------------------------------------------------------------------
//--------------------------------------gridFunctions-----------------------------------------------
//--------------------------------------------------------------------------------------------------


void populateGrid(float* X, float* Y, cl_int2 vecDim,int windowSize, int window_shift);
void gridInterpolate(float* X_ref, float* Y_ref,float* U_ref, float* V_ref, cl_int2 vecDim_ref, float* X, float* Y,float* U, float* V, cl_int2 vecDim);



//--------------------------------------------------------------------------------------------------
//--------------------------------------vectorValidation--------------------------------------------
//--------------------------------------------------------------------------------------------------

void validateVectors(cl_mem X, cl_mem Y, cl_mem U, cl_mem V, cl_mem flags, cl_int2 vecDim, cl_kernel kernel_identifyInvalidVectors, cl_kernel kernel_correctInvalidVectors, cl_command_queue queue);


#endif
