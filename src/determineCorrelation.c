#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include "libGPU_FFT.h"
#include "OpenCL_utilities.h"






// directly embed the kernel script

const char* kernelSource_complex_multiply_conjugate_norm = R"(
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


const char* kernelSource_MaxCorr = R"(

    __kernel void findMaxCorr(__global float2* input,
                            int2 inputDim,
                            int windowSize,
                            __global float* U,
                            __global float* V,
                            int2 outputDim){

    const int lid = get_local_id(0);
    const int gid[2] = {get_group_id(0),get_group_id(1)};
    
    const int startPoint = (gid[1]*windowSize)*inputDim.x + gid[0]*windowSize;
    
    __local float colMax[256];
    __local float best_i_by_row[256];
    
    //each thread will find the max in their column
    if(lid<windowSize){
      colMax[lid]=0.0;
      for(int i=0;i<windowSize;i++){
        float val = input[startPoint + i*inputDim.x + lid].x;
        if(val > colMax[lid]){
          colMax[lid] = val;
          best_i_by_row[lid] = i;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // one thread will then find the max overall
    if(lid==0){
        float maxCorr=0.0;
        int best_i = 0;
        int best_j = 0;
        for(int j=0;j<windowSize;j++){
            if (colMax[j] > maxCorr){
                best_j = j;
                best_i = best_i_by_row[j];
                maxCorr = colMax[j];
            }
        }
        if(best_i > windowSize/2){
            best_i = best_i-windowSize;
        }
        if(best_j > windowSize/2){
            best_j = best_j - windowSize;
        }
        U[gid[1]*outputDim.x + gid[0]] += -best_j;
        V[gid[1]*outputDim.x + gid[0]] += -best_i;
    }
    
    
}
)";






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

