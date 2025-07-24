#include <CL/cl.h>


const char* kernelSource_uniformTiling=R"KERN_END(

__kernel void uniform_tiling(__global float2* input, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int gid[2] = {get_group_id(0),get_group_id(1)};
int lid[2] = {get_local_id(0),get_local_id(1)};

int idx = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

output[idx_output] = input[idx];


}



)KERN_END";




void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_int2 vecDim, cl_mem output, cl_kernel kernel_uniformTiling, cl_command_queue queue){
    cl_int2 strideDim;
    strideDim.x=window_shift;strideDim.y=window_shift;
    
    cl_int2 outputDim;
    outputDim.x = vecDim.x*windowSize;
    outputDim.y = vecDim.y*windowSize;
    

    size_t localSize[2] = {windowSize,windowSize};
    size_t numGroups[2] = {vecDim.x, vecDim.y};
    size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};
    
    int idx=0;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_mem), &input); idx++;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &inputDim); idx++;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(int), &windowSize); idx++;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &strideDim); idx++;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_mem), &output); idx++;
    clSetKernelArg(kernel_uniformTiling, idx, sizeof(cl_int2), &outputDim); idx++;
    
    clEnqueueNDRangeKernel(queue, kernel_uniformTiling, 2, NULL, globalSize, localSize,0, NULL, NULL);
    clFinish(queue);
}
