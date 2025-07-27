#include "standardLibraries.h"
#include "macros.h"
#include "globalVars.h"
#include "functions.h"







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
                            int2 outputDim,
                            int subpixel){

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

        //subpixel accuracy
        float best_i_float = best_i;
        float best_j_float = best_j;
        if(maxCorr>0 && subpixel==1){
          float eps=1e-6f;
          int i_plus = (best_i+1) % windowSize; // modulus handles how index 0 is the first element, not in the middle
          int i_neg = (best_i == 0) ? (windowSize - 1) : (best_i - 1);
          int j_plus = (best_j+1) % windowSize;
          int j_neg = (best_j == 0) ? (windowSize - 1) : (best_j - 1);
          float val, val_forward,val_backward;
          float denom;
          val = input[startPoint + best_i*inputDim.x + best_j].x + eps;
          float log_val=0.0f;
          if(val > 0.0f){
            log_val = log(val);
          }
          // i
          val_forward = input[startPoint + i_plus*inputDim.x + best_j].x + eps;
          val_backward = input[startPoint + i_neg*inputDim.x + best_j].x + eps;
          if(val>0.0f && val_forward>0.0f && val_backward>0.0f){
            log_val = log(val);
            val_forward = log(val_forward);
            val_backward = log(val_backward);
            denom = (2.0f*val_backward - 4.0f*log_val + 2.0f*val_forward);
            if(fabs(denom) >1e-9f){
              best_i_float+=(val_backward-val_forward)/denom;
            }
          }
          // j
          val_forward = input[startPoint + best_i*inputDim.x + j_plus].x + eps;
          val_backward = input[startPoint + best_i*inputDim.x + j_neg].x + eps;
          if(val>0.0f && val_forward>0.0f && val_backward>0.0f){
            val_forward = log(val_forward);
            val_backward = log(val_backward);
            denom = (2.0*val_backward - 4.0*log_val + 2.0*val_forward);
            if(fabs(denom) >1e-9f){
              best_j_float+=(val_backward-val_forward)/denom;
            }
          }
        }
        if(best_i_float > windowSize/2){
            best_i_float = best_i-windowSize;
        }
        if(best_j_float > windowSize/2){
            best_j_float = best_j - windowSize;
        }
        U[gid[1]*outputDim.x + gid[0]] += -best_j_float;
        V[gid[1]*outputDim.x + gid[0]] += -best_i_float;
    }
    
    
}
)";






void FFT_corr_tiled (cl_mem input1,cl_mem input2, cl_int2 inputDim, int windowSize, OpenCL_env *env){
    cl_int err;
    // determine FFT of both inputs
    FFT2D_tiled(input1, inputDim, windowSize, 1, env->kernelFFT_1D, env->queue);
    FFT2D_tiled(input2, inputDim, windowSize, 1, env->kernelFFT_1D, env->queue);


    // multiply input1 by conjugate of input2
    // done in-place
    size_t N_windows_x = inputDim.x/windowSize;
    size_t N_windows_y = inputDim.y/windowSize;
    size_t localSize_1D = 32;
    int totalElements = (N_windows_x*N_windows_y)*(windowSize*windowSize);
    size_t numGroups_1D = ceil( (float)totalElements/(float)localSize_1D );
    size_t globalSize_1D = numGroups_1D*localSize_1D;    
    int idx=0;
    err = clSetKernelArg(env->kernelMultConj, idx, sizeof(cl_mem), &input1); idx++;      if(err!=CL_SUCCESS){printf("1 %s\n",getOpenCLErrorString(err));}
    err = clSetKernelArg(env->kernelMultConj, idx, sizeof(cl_mem), &input2); idx++;      if(err!=CL_SUCCESS){printf("2 %s\n",getOpenCLErrorString(err));}
    err = clSetKernelArg(env->kernelMultConj, idx, sizeof(int), &totalElements); idx++;          if(err!=CL_SUCCESS){printf("2 %s\n",getOpenCLErrorString(err));}
    err = clEnqueueNDRangeKernel(env->queue, env->kernelMultConj, 1, NULL, &globalSize_1D, &localSize_1D,0, NULL, NULL);      if(err!=CL_SUCCESS){printf("3 %s\n",getOpenCLErrorString(err));}
    err = clFinish(env->queue);      if(err!=CL_SUCCESS){printf("4 %s\n",getOpenCLErrorString(err));}

    // determine IFFT of input1
    FFT2D_tiled(input1, inputDim, windowSize, -1, env->kernelFFT_1D, env->queue);

}

