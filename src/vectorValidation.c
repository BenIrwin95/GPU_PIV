#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "OpenCL_utilities.h"
#include <CL/cl.h>


// Macro to find the maximum of two values
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Macro to find the minimum of two values
#define MIN(a, b) ((a) < (b) ? (a) : (b))




const char* kernelSource_vectorValidation = R"(
__kernel void identifyInvalidVectors(__global float* U,
                                     __global float* V,
                                     __global int* flags,
                                     int2 vecDim){

  int gid[2] = {get_global_id(0), get_global_id(1)};

  if(gid[0]<vecDim.x && gid[1]<vecDim.y){
    int idx = gid[1]*vecDim.x + gid[0];
    float2 avg;
    avg.x=0.0;avg.y=0.0f;
    int count=0;
    for(int i=-1;i<1;i++){
      for(int j=-1;j<1;j++){
        if(i==0 && j==0){continue;}//only want neighbours
        int i_=gid[1]+i;
        int j_=gid[0]+j;
        // Check if the potential neighbor is within the array bounds
        if (i_ >= 0 && i_ < vecDim.y && j_ >= 0 && j_ < vecDim.x){
          //sum_neighbors += input_array[get_1d_index(neighbor_row, neighbor_col, cols)];
          avg.x += U[i_*vecDim.x + j_];
          avg.y += V[i_*vecDim.x + j_];
          count++;
        }
      }
    }
    avg.x=avg.x/count;avg.y=avg.y/count;
    float criterion = sqrt(pow(avg.x - U[idx],2) + pow(avg.y - V[idx],2));
    criterion = criterion/sqrt(pow(avg.x,2) + pow(avg.y,2));
    if(criterion > 0.1){
      flags[idx] = 1;
    } else {
      flags[idx] = 0;
    }
  }
}



__kernel void correctInvalidVectors(__global float* X,
                                    __global float* Y,
                                    __global float* U,
                                    __global float* V,
                                    __global int* flags,
                                    int2 vecDim){
  int gid[2] = {get_global_id(0), get_global_id(1)};
  if(gid[0]<vecDim.x && gid[1]<vecDim.y){
    int idx = gid[1]*vecDim.x + gid[0];
    if(flags[idx]==1){ // marked for replacement
      int srcDist = 4; // how far we want to look for points to interpolate from
      int i_start, i_end, j_start, j_end;
      j_start = gid[0]-srcDist;                     if(j_start<0){j_start=0;};
      j_end =gid[0]+srcDist;                        if(j_end>vecDim.x){j_end=vecDim.x;};
      i_start = gid[1]-srcDist;            if(i_start<0){i_start=0;};
      i_end = gid[1]+srcDist;              if(i_end>vecDim.y){i_end=vecDim.y;};
      // iterate through the points we will interpolate from
      float U_new=0.0f;
      float V_new=0.0f;
      float sum_weights=0.0f;
      for(int i=i_start;i<i_end;i++){
        for(int j=j_start;j<j_end;j++){
          int idx_local = i*vecDim.x+j;
          if(flags[idx_local]==0){
            float d2 = pow(X[idx]-X[idx_local],2) + pow(Y[idx]-Y[idx_local],2);
            float weight = 1.0/d2;
            sum_weights+=weight;
            U_new += weight*U[idx_local];
            V_new += weight*V[idx_local];
          }
        }
      }

      U[idx] = U_new/sum_weights;
      V[idx] = V_new/sum_weights;


    }
  }
}

)";




void validateVectors(cl_mem X, cl_mem Y, cl_mem U, cl_mem V, cl_mem flags, cl_int2 vecDim, cl_kernel kernel_identifyInvalidVectors, cl_kernel kernel_correctInvalidVectors, cl_command_queue queue){
  cl_int err;
  size_t localSize_1D = 5;
  size_t totalElements[2] = {vecDim.x, vecDim.y};
  size_t numGroups[2];
  numGroups[0] = ceil( (float)totalElements[0]/localSize_1D );
  numGroups[1] = ceil( (float)totalElements[1]/localSize_1D );
  size_t globalSize[2] = {numGroups[0]*localSize_1D,numGroups[1]*localSize_1D};
  size_t localSize[2] = {localSize_1D, localSize_1D};
  // global size has to be a multiple of the local size, hence this somewhat convoluted setup
  // will filter the innecessary threads kernel-side

  // identify vectors that need correcting
  int idx=0;
  err = clSetKernelArg(kernel_identifyInvalidVectors, idx, sizeof(cl_mem), &U); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_identifyInvalidVectors, idx, sizeof(cl_mem), &V); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_identifyInvalidVectors, idx, sizeof(cl_mem), &flags); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_identifyInvalidVectors, idx, sizeof(cl_int2), &vecDim); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clEnqueueNDRangeKernel(queue, kernel_identifyInvalidVectors, 2, NULL, globalSize, localSize,0, NULL, NULL);      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clFinish(queue);      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}

  // correct the vectors
  idx=0;
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_mem), &X); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_mem), &Y); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_mem), &U); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_mem), &V); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_mem), &flags); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clSetKernelArg(kernel_correctInvalidVectors, idx, sizeof(cl_int2), &vecDim); idx++;      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clEnqueueNDRangeKernel(queue, kernel_correctInvalidVectors, 2, NULL, globalSize, localSize,0, NULL, NULL);      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
  err = clFinish(queue);      if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}

}


