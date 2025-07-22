#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>



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



// directly embed the kernel script

const char* kernelSource_FFT_functions = R"(
float2 multiply_complex(float2 a, float2 b){
    float2 out;
    out.x = a.x*b.x - a.y*b.y;
    out.y = a.x*b.y + a.y*b.x;
    return out;
}


__kernel void FFT_1D(__global float2* input, int2 inputDim, int2 strideDim, int step, int N, int dir){
    // input: the 2D input array
    // inputDim: (width, height) of the inptu array
    // strideDim: (gid[1]*strideDim.y)*inputWidth + gid[0]*strideDim.x ==> sets the starting point of FFT
    // step: when iterating, how many elements until the next consequtive element we want
    // N: size of the sample the FFT is being performed on
    // dir: 1 = forward, dir: -1 = inverse

    const int gid[2] = {get_group_id(0), get_group_id(1)};
    const int lid = get_local_id(0);
    // assumes the kernel will be assigned N local threads




    // load relevant data to __local
    const int inputWidth = inputDim.x;
    const int startPoint = (gid[1]*strideDim.y)*inputWidth + gid[0]*strideDim.x;
    __local float2 inputLocal[256];
    __local float2 outputLocal[256];
    inputLocal[lid].x = input[ startPoint + lid*step ].x;
    inputLocal[lid].y = input[ startPoint + lid*step ].y;
    barrier(CLK_LOCAL_MEM_FENCE);


    float exp_factor;
    float scalingFactor;
    if(dir==1.0){
        exp_factor = -1.0;
        scalingFactor = 1.0;
    }
    if(dir==-1.0){
        exp_factor = 1.0;
        scalingFactor = 1.0/(float)N;
    }


    // perform Cooley-Tukey FFT
    const int M = N/2;
    if(lid<M){
        int k = lid;
        float2 Ek; // even index elements
        float2 Ok; // odd index elements
        Ek.x=0.0;Ek.y=0.0;
        Ok.x=0.0;Ok.y=0.0;
        for(int m=0;m<M;m++){
            int idx_even = 2*m;
            int idx_odd = 2*m+1;
            float arg = exp_factor*2.0*M_PI*m*k/M;
            float2 EXP;
            EXP.x = cos(arg);EXP.y = sin(arg);
            float2 x_even = inputLocal[idx_even];
            float2 x_odd = inputLocal[idx_odd];
            float2 temp;
            temp = multiply_complex(x_even, EXP);
            Ek.x += temp.x;Ek.y += temp.y;
            temp = multiply_complex(x_odd, EXP);
            Ok.x += temp.x;Ok.y += temp.y;
        }
        float arg = exp_factor*2.0*M_PI*k/N;
        float2 EXP;EXP.x = cos(arg);EXP.y = sin(arg);
        float2 temp;
        temp = multiply_complex(Ok, EXP);
        outputLocal[k].x = (Ek.x + temp.x)*scalingFactor;
        outputLocal[k].y = (Ek.y + temp.y)*scalingFactor;
        outputLocal[k+M].x = (Ek.x - temp.x)*scalingFactor;
        outputLocal[k+M].y = (Ek.y - temp.y)*scalingFactor;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // copy data back to global
    input[ startPoint + lid*step ].x = outputLocal[lid].x;
    input[ startPoint + lid*step ].y = outputLocal[lid].y;

}



__kernel void complex_multiply_conjugate_norm(__global float2* A,
                                              __global float2* B,
                                              int N)
{
    // this will be done in place and overwrite A
    const int gid = get_global_id(0);

    if(gid<N){
        float2 a = A[gid];
        float2 b = B[gid];
        float2 c;

        // conjugate b
        b.y = -1.0*b.y;

        // multiply
        c.x = a.x*b.x - a.y*b.y;
        c.y = a.x*b.y + a.y*b.x;
        float mag_sq = c.x * c.x + c.y * c.y;
        if(mag_sq>0.0f){
          float inv_mag = rsqrt(mag_sq);
          c.x = c.x * inv_mag;
          c.y = c.y * inv_mag;
        } else {
          c.x=0.0f;
          c.y=0.0f;
        }
        

        A[gid] = c;
    }

}
)";







void FFT2D_tiled (cl_mem inputGPU, cl_int2 inputDim, int windowSize, int dir, cl_kernel kernelFFT_1D, cl_command_queue queue){
    cl_int err;
    size_t N_windows_x = inputDim.x/windowSize;
    size_t N_windows_y = inputDim.y/windowSize;
    size_t localSize[2];
    size_t numGroups[2];
    size_t globalSize[2];
    cl_int2 strideDim; // how many elements we want to move in x and y for each group
    int step; // number of elements to move in input, to get the next consecutive element we need to use
    int N; // number of elements we will perform FFT over
    N=windowSize;
    
    localSize[0] = windowSize;
    localSize[1] = 1;
    
    // first perform row-wise operations
    numGroups[0] = N_windows_x;
    numGroups[1] = inputDim.y;
    globalSize[0] = localSize[0]*numGroups[0];
    globalSize[1] = localSize[1]*numGroups[1];
    strideDim.x=windowSize;
    strideDim.y=1;
    step=1;
  
    // set arguments for kernel function
    int idx=0;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_mem), &inputGPU); idx++;        if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &inputDim); idx++;       if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &strideDim); idx++;      if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &step); idx++;               if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &N); idx++;                  if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &dir); idx++;                if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    // Execute the kernel over the entire range of the data set
    err=clEnqueueNDRangeKernel(queue, kernelFFT_1D, 2, NULL, globalSize, localSize,0, NULL, NULL); if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err = clFinish(queue);


    // next perform column-wise operations
    // columnwise
    numGroups[0] = inputDim.x;
    numGroups[1] = N_windows_y;
    globalSize[0] = localSize[0]*numGroups[0];
    globalSize[1] = localSize[1]*numGroups[1];
    strideDim.x=1;
    strideDim.y=windowSize;
    step=inputDim.x;
    err=clSetKernelArg(kernelFFT_1D, 2, sizeof(cl_int2), &strideDim);      if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clSetKernelArg(kernelFFT_1D, 3, sizeof(int), &step);               if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err=clEnqueueNDRangeKernel(queue, kernelFFT_1D, 2, NULL, globalSize, localSize,0, NULL, NULL);      if(err!=CL_SUCCESS){printf("%s\n",getOpenCLErrorString(err));}
    err = clFinish(queue);
};


void FFT_corr_tiled (cl_mem input1,cl_mem input2, cl_int2 inputDim, int windowSize, cl_kernel kernelFFT_1D, cl_kernel kernelMultiplyConj, cl_command_queue queue){
    cl_int err;
    // determine FFT of both inputs
    FFT2D_tiled(input1, inputDim, windowSize, 1, kernelFFT_1D, queue);
    FFT2D_tiled(input2, inputDim, windowSize, 1, kernelFFT_1D, queue);


    // multiply input1 by conjugate of input2
    // done in-place
    size_t N_windows_x = inputDim.x/windowSize;
    size_t N_windows_y = inputDim.y/windowSize;
    size_t localSize_1D = 32;
    int totalElements = (N_windows_x*N_windows_y)*(windowSize*windowSize);
    size_t numGroups_1D = ceil( (float)totalElements/(float)localSize_1D );
    size_t globalSize_1D = numGroups_1D*localSize_1D;    
    int idx=0;
    err = clSetKernelArg(kernelMultiplyConj, idx, sizeof(cl_mem), &input1); idx++;      if(err!=CL_SUCCESS){printf("1 %s\n",getOpenCLErrorString(err));}
    err = clSetKernelArg(kernelMultiplyConj, idx, sizeof(cl_mem), &input2); idx++;      if(err!=CL_SUCCESS){printf("2 %s\n",getOpenCLErrorString(err));}
    err = clSetKernelArg(kernelMultiplyConj, idx, sizeof(int), &totalElements); idx++;          if(err!=CL_SUCCESS){printf("2 %s\n",getOpenCLErrorString(err));}
    err = clEnqueueNDRangeKernel(queue, kernelMultiplyConj, 1, NULL, &globalSize_1D, &localSize_1D,0, NULL, NULL);      if(err!=CL_SUCCESS){printf("3 %s\n",getOpenCLErrorString(err));}
    err = clFinish(queue);      if(err!=CL_SUCCESS){printf("4 %s\n",getOpenCLErrorString(err));}

    // determine IFFT of input1
    FFT2D_tiled(input1, inputDim, windowSize, -1, kernelFFT_1D, queue);

}

