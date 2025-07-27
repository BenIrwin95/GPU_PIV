#ifndef GLOBAL_VARS_H
#define GLOBAL_VARS_H
#include "standardLibraries.h"


typedef struct OpenCL_env {
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_command_queue queueNonBlocking;  // command queue (non-blocking --> useful under certain scenarios)
    cl_program program;                 // program
    // the various kernels this code will be creating
    cl_kernel kernelMultConj;
    cl_kernel kernelMaxCorr;
    cl_kernel kernelFFT_1D;
    cl_kernel kernel_uniformTiling;
    cl_kernel kernel_offsetTiling;
    cl_kernel kernel_identifyInvalidVectors;
    cl_kernel kernel_correctInvalidVectors;
} OpenCL_env;


typedef struct PIVdata {
    int N_pass;
    float** X_passes;
    float** Y_passes;
    float** U_passes;
    float** V_passes;
    cl_int2* vecDim_passes;
} PIVdata;



//--------------------------------------------------------------------------------------------------
//--------------------------------------dataArrangement---------------------------------------------
//--------------------------------------------------------------------------------------------------
extern const char* kernelSource_uniformTiling;


//--------------------------------------------------------------------------------------------------
//--------------------------------------determineCorrelation----------------------------------------
//--------------------------------------------------------------------------------------------------

extern const char* kernelSource_complex_multiply_conjugate_norm;
extern const char* kernelSource_MaxCorr;


//--------------------------------------------------------------------------------------------------
//--------------------------------------vectorValidation--------------------------------------------
//--------------------------------------------------------------------------------------------------

extern const char* kernelSource_vectorValidation;
#endif
