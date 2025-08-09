#include "standardHeader.hpp"



const std::string kernelSource_dataArrangement=R"KERN_END(

__kernel void uniform_tiling(__global float2* input, int2 inputDim, int N, int2 strideDim,__global float2* output, int2 outputDim){

int gid[2] = {get_group_id(0),get_group_id(1)};
int lid[2] = {get_local_id(0),get_local_id(1)};

int idx = (gid[1]*strideDim.y + lid[1])*inputDim.x + (gid[0]*strideDim.x + lid[0]);

int idx_output = (gid[1]*N + lid[1])*outputDim.x + (gid[0]*N + lid[0]);

output[idx_output] = input[idx];


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


/*
void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_int2 vecDim, cl_mem output, OpenCL_env *env){
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
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(cl_mem), &input); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(cl_int2), &inputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(int), &windowSize); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(cl_int2), &strideDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(cl_mem), &output); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_uniformTiling, idx, sizeof(cl_int2), &outputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}

    err=clEnqueueNDRangeKernel(env->queue, env->kernel_uniformTiling, 2, NULL, globalSize, localSize,0, NULL, NULL);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clFinish(env->queue);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
}
*/

/*
cl_int uploadImage_and_convert_to_complex(ImageData& im, OpenCL_env& env, cl::Buffer& buffer, cl::Buffer& buffer_complex){
    cl_int err;

    const void* host_ptr = std::visit([](const auto& vec) -> const void* {return vec.data();}, im.pixelData);
    if (!host_ptr) {
        // Handle error: host pointer is null
        return CL_INVALID_HOST_PTR; // A custom or predefined error code
    }
    try{
        err = env.queue.enqueueWriteBuffer( buffer, CL_TRUE, 0, im.sizeBytes, host_ptr);
    } catch (cl::Error& e) {
        std::cerr << "Error uploading to GPU" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }

    cl_int N = im.width*im.height;
    try {
        err = env.kernel_convert_im_to_complex.setArg(0, buffer);
    } catch (cl::Error& e) {
        std::cerr << "Error setting kernel argument 0" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    try {
        err = env.kernel_convert_im_to_complex.setArg(1, buffer_complex);
    } catch (cl::Error& e) {
        std::cerr << "Error setting kernel argument 1" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    try {
        err = env.kernel_convert_im_to_complex.setArg(2, sizeof(cl_int),&N);
    } catch (cl::Error& e) {
        std::cerr << "Error setting kernel argument 2" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_convert_im_to_complex, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_convert_im_to_complex" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }

    return CL_SUCCESS;
}*/
