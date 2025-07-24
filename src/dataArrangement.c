#include <CL/cl.h>
#include "OpenCL_utilities.h"




const char* kernelSource_uniformTiling=R"KERN_END(

__kernel void uniform_tiling(__global float2* input, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int gid[2] = {get_group_id(0),get_group_id(1)};
int lid[2] = {get_local_id(0),get_local_id(1)};

int idx = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

output[idx_output] = input[idx];


}


__kernel void offset_tiling(__global float2* input, int2 inputDim,
                            int N, int2 strideDim,
                            __global float* U, __global float* V, int2 vecDim,
                            __global float2* output, int2 outputDim){


    int gid[2] = {get_group_id(0),get_group_id(1)};
    int lid[2] = {get_local_id(0),get_local_id(1)};

    __local float U_local;
    __local float V_local;
    U_local = U[gid[1]*vecDim.x + gid[0]];
    V_local = V[gid[1]*vecDim.x + gid[0]];


    int2 idx;
     // bottom left corner
    idx.x = gid[0]*strideDim.x;
    idx.y = gid[1]*strideDim.y;

    // U_local = 0;
    // V_local = 0;
    // checks to prevent window going off edge
    if(lid[0]==0 && lid[1]==0){
        if(idx.x + U_local + N>inputDim.x){
            U_local -= idx.x+U_local+N;
            U[gid[1]*vecDim.x + gid[0]] = U_local;
        }
        if(idx.y + V_local + N>inputDim.y){
            V_local -= idx.y+U_local+N;
            V[gid[1]*vecDim.x + gid[0]] = V_local;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    idx.x+=U_local + lid[0];
    idx.y+=V_local + lid[1];


    int idx_input = idx.y*inputDim.x + idx.x;

    int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

    output[idx_output] = input[idx_input];

}



)KERN_END";




void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_int2 vecDim, cl_mem output, cl_kernel kernel_uniformTiling, cl_command_queue queue){
    cl_int err;

    cl_int2 strideDim;
    strideDim.x=window_shift;strideDim.y=window_shift;
    
    cl_int2 outputDim;
    outputDim.x = vecDim.x*windowSize;
    outputDim.y = vecDim.y*windowSize;
    

    size_t localSize[2] = {windowSize,windowSize};
    size_t numGroups[2] = {vecDim.x, vecDim.y};
    size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};
    
    int idx=0;
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_mem), &input); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &inputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(int), &windowSize); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &strideDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_mem), &output); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &outputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    
    err=clEnqueueNDRangeKernel(queue, kernel_uniformTiling, 2, NULL, globalSize, localSize,0, NULL, NULL);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clFinish(queue);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
}




// function to call offset_tiling kernel
void offset_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_mem U_GPU, cl_mem V_GPU, cl_int2 vecDim, cl_mem output, cl_kernel kernel_offsetTiling, cl_command_queue queue){
    cl_int err;

    cl_int2 strideDim;
    strideDim.x=window_shift;strideDim.y=window_shift;

    cl_int2 outputDim;
    outputDim.x = vecDim.x*windowSize;
    outputDim.y = vecDim.y*windowSize;


    size_t localSize[2] = {windowSize,windowSize};
    size_t numGroups[2] = {vecDim.x, vecDim.y};
    size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};

    int idx=0;
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_mem), &input); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_int2), &inputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(int), &windowSize); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_int2), &strideDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_mem), &U_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_mem), &V_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_int2), &vecDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_mem), &output); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(kernel_offsetTiling, idx, sizeof(cl_int2), &outputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}

    err=clEnqueueNDRangeKernel(queue, kernel_offsetTiling, 2, NULL, globalSize, localSize,0, NULL, NULL);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clFinish(queue);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
}
