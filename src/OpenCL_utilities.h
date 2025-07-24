#ifndef OPENCL_UTILITIES_H
#define OPENCL_UTILITIES_H
#include <CL/cl.h>

const char* getOpenCLErrorString(cl_int err);

cl_int initialise_OpenCL(cl_platform_id *platform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue, cl_command_queue *queueNonBlocking, cl_program *program,cl_kernel *kernelFFT_1D, cl_kernel *kernelMultConj, cl_kernel *kernelMaxCorr);

#endif
