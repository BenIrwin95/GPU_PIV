#include "standardHeader.hpp"



const std::string kernelSource_dataArrangement=R"KERN_END(

__kernel void uniform_tiling(__global float2* input, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int global_id[2] = {get_global_id(0), get_global_id(1)};
int gid[2] = {floor((float)global_id[0]/N), floor((float)global_id[1]/N)};
int lid[2] = {global_id[0] % N, global_id[1] % N};

int idx = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

output[idx_output] = input[idx];



}


__kernel void warped_tiling(__global float2* input, __global float* offsets_x, __global float* offsets_y, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int global_id[2] = {get_global_id(0), get_global_id(1)};
int gid[2] = {floor((float)global_id[0]/N), floor((float)global_id[1]/N)};
int lid[2] = {global_id[0] % N, global_id[1] % N};

int idx_og = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int2 idx;
idx.x = gid[0]*strideDim.x + lid[0];
idx.y = gid[1]*strideDim.y + lid[1];

idx.x += (int)offsets_x[idx_og];
idx.y += (int)offsets_y[idx_og];



int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);
if(idx.x < 0 || idx.x >= inputDim.x || idx.y < 0 || idx.y >= inputDim.y){
    output[idx_output]=0.0f;
} else {
    output[idx_output] = input[idx.y*inputDim.x + idx.x];
}

}



__kernel void detrend_window(__global float2* input, int2 inputDim, int N, __local float* rowSum){
    
   
    int global_id[2] = {get_global_id(0), get_global_id(1)};
    int gid[2] = {floor((float)global_id[0]/N), floor((float)global_id[1]/N)};
    int lid[2] = {global_id[0] % N, global_id[1] % N};

    int idx_corner = (gid[1]*N)*inputDim.x + (gid[0]*N);

    // summation of all pixels
    float fullSum=0.0f;
    if(lid[0]==0){
        // sum each row of window
        rowSum[lid[1]]=0.0f;
        for(int i=0;i<N;i++){
            rowSum[lid[1]] += input[idx_corner + lid[1]*inputDim.x + i].x; // pre-FFT we know the image will only exist in the x component
        }
        // now sum the row sums
        if(lid[1]==0){
            for(int i=0;i<N;i++){
                fullSum+=rowSum[i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int idx_output = (gid[1]*N + lid[1])*inputDim.x + (gid[0]*N + lid[0]);
    float output = input[idx_output].x - fullSum/(N*N);
    if(output < 0){
        output = 0;
    }
    input[idx_output].x = output;

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


    //cl::NDRange local(windowSize, windowSize);
    cl::NDRange local(8, 8);
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


    try {err = env.kernel_warped_tiling.setArg(0, input);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(1, env.imageShifts_x);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(2, env.imageShifts_y);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(3, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(4, sizeof(int), &windowSize);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(5, sizeof(cl_int2), &strideDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(6, output);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_warped_tiling.setArg(7, sizeof(cl_int2), &outputDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    //cl::NDRange local(windowSize, windowSize);
    cl::NDRange local(8, 8);
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

//__kernel void detrend_window(__global float2* input, int2 inputDim, int N, __local float* rowSum)
cl_int detrend_windows(cl::Buffer& input, cl_int2 inputDim, int windowSize, cl_int2 arrSize, OpenCL_env& env){
    cl_int err = CL_SUCCESS;

    //cl::NDRange local(windowSize, windowSize);
    cl::NDRange local(8, 8);
    cl::NDRange global(arrSize.s[0]*windowSize, arrSize.s[1]*windowSize);

    try {err = env.kernel_detrend_window.setArg(0, input);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_detrend_window.setArg(1, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_detrend_window.setArg(2, sizeof(int), &windowSize);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_detrend_window.setArg(3, windowSize*sizeof(float), NULL);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_detrend_window, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_detrend_window" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}
