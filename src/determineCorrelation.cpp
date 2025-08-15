#include "standardHeader.hpp"


const std::string kernelSource_determineCorrelation=R"KERN_END(
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
        /*
        if(mag_sq>0.0f){
          float inv_mag = rsqrt(mag_sq);
          c.x = c.x * inv_mag;
          c.y = c.y * inv_mag;
        } else {
          c.x=0.0f;
          c.y=0.0f;
        }*/


        A[gid] = c;
    }

}


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
            // FFT based correlation finding has an inherent bias towards 0 and thus a correction needs applying
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

)KERN_END";


cl_int FFT_corr_tiled(cl::Buffer& input1, cl::Buffer& input2, cl_int2 inputDim, int windowSize, OpenCL_env& env){
    cl_int err = CL_SUCCESS;

    // find FFT of both inputs
    err = FFT2D_tiled(input1, inputDim, windowSize, 1, env);if(err!=CL_SUCCESS){CHECK_CL_ERROR(err);return err;}
    err = FFT2D_tiled(input2, inputDim, windowSize, 1, env);if(err!=CL_SUCCESS){CHECK_CL_ERROR(err);return err;}


    // multiply input1 by conjugate of input2
    // done in-place on input1
    size_t N_windows_x = inputDim.s[0]/windowSize;
    size_t N_windows_y = inputDim.s[1]/windowSize;
    size_t localSize = 64;
    int totalElements = (N_windows_x*N_windows_y)*(windowSize*windowSize);
    size_t numGroups = ceil( (float)totalElements/(float)localSize );
    size_t globalSize = numGroups*localSize;
    try {err = env.kernel_complex_multiply_conjugate_norm.setArg(0, input1);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_complex_multiply_conjugate_norm.setArg(1, input2);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_complex_multiply_conjugate_norm.setArg(2, sizeof(int), &totalElements);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    cl::NDRange local(localSize);
    cl::NDRange global(globalSize);
    try{env.queue.enqueueNDRangeKernel(env.kernel_complex_multiply_conjugate_norm, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_complex_multiply_conjugate_norm" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();


    // compute inverse FFT for input1
    err = FFT2D_tiled(input1, inputDim, windowSize, -1, env);if(err!=CL_SUCCESS){CHECK_CL_ERROR(err);return err;}

    return err;
}



cl_int find_max_corr(cl::Buffer& input, cl_int2 inputDim, int windowSize, cl::Buffer& outputU, cl::Buffer& outputV, cl_int2 arrSize, int activate_subpixel, OpenCL_env& env){
    cl_int err=CL_SUCCESS;
    size_t localSize[2] = {static_cast<size_t>(windowSize),1};
    size_t numGroups[2] = {static_cast<size_t>(arrSize.s[0]), static_cast<size_t>(arrSize.s[1])};
    size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};

    try {err = env.kernel_findMaxCorr.setArg(0, input);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(1, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(2, sizeof(int), &windowSize);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(3, outputU);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(4, outputV);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(5, sizeof(cl_int2), &arrSize);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_findMaxCorr.setArg(6, sizeof(int), &activate_subpixel);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    cl::NDRange local(localSize[0],localSize[1]);
    cl::NDRange global(globalSize[0],globalSize[1]);
    try{env.queue.enqueueNDRangeKernel(env.kernel_findMaxCorr, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_findMaxCorr" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();


    return err;
}


