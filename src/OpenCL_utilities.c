#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "libGPU_FFT.h"
#include "determineCorrelation.h"
#include "dataArrangement.h"



const char* getOpenCLErrorString(cl_int err) {
    switch (err) {
        // Run-time Errors
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // Compile-time Errors
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";

        default:                                    return "Unknown OpenCL Error";
    }
}

#define ERROR_MSG_OPENCL(err) fprintf(stderr,"ERROR: %s - %s:%d:%s:", getOpenCLErrorString(err), __FILE__, __LINE__, __func__)



cl_int initialise_OpenCL(cl_platform_id *platform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue, cl_command_queue *queueNonBlocking, cl_program *program, 
                          cl_kernel *kernelFFT_1D, cl_kernel *kernelMultConj, cl_kernel *kernelMaxCorr, cl_kernel *kernel_uniformTiling){
    cl_int err;
    // Bind to platform
    err = clGetPlatformIDs(1, platform, NULL);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    // Get ID for the device
    err = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 1, device_id, NULL);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    // Create a context
    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    // Create a command queue
    *queue = clCreateCommandQueue(*context, *device_id, 0, &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}


    cl_queue_properties non_blocking_properties[] = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        0 // Must terminate with 0
    };
    *queueNonBlocking = clCreateCommandQueueWithProperties(*context, *device_id, non_blocking_properties, &err);

    // Create the compute program from the source buffer
    const char* kernel_sources[] = { kernelSource_complexMaths, kernelSource_FFT_1D, kernelSource_complex_multiply_conjugate_norm, kernelSource_MaxCorr, kernelSource_uniformTiling};
    *program = clCreateProgramWithSource(*context, 4, kernel_sources, NULL, &err);
    //free(kernel_sources);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    // Build the program executable
    err = clBuildProgram(*program, 1, device_id, NULL, NULL, NULL);
    if(err != CL_SUCCESS){ // handling errors when compiling the kernel
        size_t log_size;
        clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *) malloc(log_size + 1); // +1 for null terminator
        // Get the log
        clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0'; // Null-terminate
        fprintf(stderr, "Kernel Build Error:\n%s\n", build_log);
        free(build_log);
    }
    // Create the compute kernel in the program
    *kernelMultConj = clCreateKernel(*program, "complex_multiply_conjugate_norm", &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    *kernelMaxCorr = clCreateKernel(*program, "findMaxCorr", &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    *kernelFFT_1D = clCreateKernel(*program, "FFT_1D", &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    *kernel_uniformTiling = clCreateKernel(*program, "uniform_tiling", &err);
    if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);return err;}
    return CL_SUCCESS;
}
