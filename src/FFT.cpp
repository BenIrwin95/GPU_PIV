#include "standardHeader.hpp"



const std::string kernelSource_FFT = R"KERN_END(


    // function to return an integer after inversing its bits
    int reverse_bits(int n, int num_bits) {
        int reversed_n = 0;
        for (int i = 0; i < num_bits; i++) {
            reversed_n = (reversed_n << 1) | (n & 1);
            n >>= 1;
        }
        return reversed_n;
    }

    void bit_reverse_permutation(float2* input, int N, int lid){
        // designed for use in a workgroup where max(lid)>=N
        int num_bits = (int)log2((float)N);
        if(lid<N){
            int i = lid;
            int j = reverse_bits(i,num_bits);
            if(i < j){ // prevents doing swap twice and thus undoing swap
                float2 temp = input[i];
                input[i] = input[j];
                input[j] = temp;
            }
        }
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


        // load relevant data to __local
        const int inputWidth = inputDim.x;
        const int startPoint = (gid[1]*strideDim.y)*inputWidth + gid[0]*strideDim.x;
        __local float2 inputLocal[256];
        inputLocal[lid].x = input[ startPoint + lid*step ].x;
        inputLocal[lid].y = input[ startPoint + lid*step ].y;
        barrier(CLK_LOCAL_MEM_FENCE);


        // rearrange data with bit reversal bit_reverse_permutation
        bit_reverse_permutation(inputLocal, N, lid);
        barrier(CLK_LOCAL_MEM_FENCE);

        // define some modifiers depending on whether we are doing forward or backward fft
        float exp_factor;
        if(dir==1){
            exp_factor = -1.0;
        }
        if(dir==-1){
            exp_factor = 1.0;
        }


        // FFT
        int N_pass = log2((float)N);
        for(int i=0;i<N_pass;i++){
            int N_local = (int)pow(2.0f,i+1);
            int M=N_local/2;
            // figure out which strtPt this lid corresponds with
            int strtPt=N_local*floor((float)lid/N_local);
            // figure out which element within the group it corresponds with
            int k = lid % N_local;
            if(k<M){
                float arg = exp_factor*2.0*M_PI*k/N_local;
                float2 EXP;
                EXP.x=cos(arg);EXP.y=sin(arg);
                float2 val1 = inputLocal[strtPt+k];
                float2 val2 = multiply_complex(EXP, inputLocal[strtPt+k+M]);
                inputLocal[strtPt+k] = add_complex(val1,val2);
                inputLocal[strtPt+k+M] = subtract_complex(val1,val2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(dir == -1){
            inputLocal[lid] = scale_complex(inputLocal[lid], 1.0f / (float)N); // Apply 1/N here
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // copy data back to global
        input[ startPoint + lid*step ].x = inputLocal[lid].x;
        input[ startPoint + lid*step ].y = inputLocal[lid].y;

    }

)KERN_END";




cl_int FFT2D_tiled(cl::Buffer& input, cl_int2 inputDim, int windowSize, int dir, OpenCL_env& env){
    cl_int err=CL_SUCCESS;

    size_t N_windows_x = inputDim.x/windowSize;
    size_t N_windows_y = inputDim.y/windowSize;
    size_t localSize[2];
    size_t numGroups[2];
    size_t globalSize[2];
    cl_int2 strideDim; // how many elements we want to move in x and y for each group
    int step; // number of elements to move in input, to get the next consecutive element we need to use
    int N; // number of elements we will perform FFT over
    N=windowSize;

    // on the local level, we only need a 1D localSize
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
    try {err = env.kernel_FFT_1D.setArg(0, input);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(1, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_mem), &inputGPU); idx++;
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &inputDim); idx++;
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &strideDim); idx++;
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &step); idx++;
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &N); idx++;
    // err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &dir); idx++;

    return err;
}

/*
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
    int idx;
    idx=0;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_mem), &inputGPU); idx++;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &inputDim); idx++;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(cl_int2), &strideDim); idx++;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &step); idx++;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &N); idx++;
    err=clSetKernelArg(kernelFFT_1D, idx, sizeof(int), &dir); idx++;
    // Execute the kernel over the entire range of the data set
    err=clEnqueueNDRangeKernel(queue, kernelFFT_1D, 2, NULL, globalSize, localSize,0, NULL, NULL);
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
    err=clSetKernelArg(kernelFFT_1D, 2, sizeof(cl_int2), &strideDim); idx++;
    err=clSetKernelArg(kernelFFT_1D, 3, sizeof(int), &step); idx++;
    err=clEnqueueNDRangeKernel(queue, kernelFFT_1D, 2, NULL, globalSize, localSize,0, NULL, NULL);
    err = clFinish(queue);

};*/
