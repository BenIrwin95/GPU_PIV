#include "standardHeader.hpp"



const std::string kernelSource_dataArrangement=R"KERN_END(

__kernel void uniform_tiling(__global float2* input, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int gid[2] = {get_group_id(0),get_group_id(1)};
int lid[2] = {get_local_id(0),get_local_id(1)};

int idx = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

output[idx_output] = input[idx];


}


__kernel void warped_tiling(__global float2* input, __global int2* offsets, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int gid[2] = {get_group_id(0),get_group_id(1)};
int lid[2] = {get_local_id(0),get_local_id(1)};

int idx_og = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int2 idx;
idx.x = gid[0]*strideDim.x + lid[0];
idx.y = gid[1]*strideDim.y + lid[1];

idx.x += offsets[idx_og].x;
idx.y += offsets[idx_og].y;



int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);
if(idx.x < 0 || idx.x >= inputDim.x || idx.y < 0 || idx.y >= inputDim.y){
    output[idx_output]=0.0f;
} else {
    output[idx_output] = input[idx.y*inputDim.x + idx.x];
}

}

)KERN_END";


cl_int uniformly_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env){
    cl_int err = CL_SUCCESS;

    cl_int2 strideDim;
    strideDim.s[0]=window_shift;strideDim.s[1]=window_shift;

    cl_int2 outputDim;
    outputDim.s[0] = arrSize.s[0]*windowSize;
    outputDim.s[1] = arrSize.s[1]*windowSize;


    try {err = env.kernel_uniform_tiling.setArg(0, input);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_uniform_tiling.setArg(1, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_uniform_tiling.setArg(2, sizeof(int), &windowSize);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_uniform_tiling.setArg(3, sizeof(cl_int2), &strideDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_uniform_tiling.setArg(4, output);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_uniform_tiling.setArg(5, sizeof(cl_int2), &outputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}


    cl::NDRange local(windowSize, windowSize);
    cl::NDRange global(arrSize.s[0]*windowSize, arrSize.s[1]*windowSize);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_uniform_tiling, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_uniform_tiling" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();
    return err;
}



cl_int warped_tile_data(cl::Buffer& input, cl_int2 inputDim, cl::Buffer& output, int windowSize, int window_shift, cl_int2 arrSize, OpenCL_env& env){
    cl_int err = CL_SUCCESS;

    cl_int2 strideDim;
    strideDim.s[0]=window_shift;strideDim.s[1]=window_shift;

    cl_int2 outputDim;
    outputDim.s[0] = arrSize.s[0]*windowSize;
    outputDim.s[1] = arrSize.s[1]*windowSize;


    try {err = env.kernel_warped_tiling.setArg(0, input);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(1, env.imageShifts);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(2, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(3, sizeof(int), &windowSize);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(4, sizeof(cl_int2), &strideDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(5, output);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(6, sizeof(cl_int2), &outputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 6" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}


    cl::NDRange local(windowSize, windowSize);
    cl::NDRange global(arrSize.s[0]*windowSize, arrSize.s[1]*windowSize);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_warped_tiling, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_warped_tiling" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();
    return err;
}

