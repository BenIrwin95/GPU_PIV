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
    cl::NDRange local;
    cl::NDRange global;

    size_t N_windows_x = inputDim.s[0]/windowSize;
    size_t N_windows_y = inputDim.s[1]/windowSize;
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
    numGroups[1] = inputDim.s[1];
    globalSize[0] = localSize[0]*numGroups[0];
    globalSize[1] = localSize[1]*numGroups[1];
    strideDim.s[0]=windowSize;
    strideDim.s[1]=1;
    step=1;
    try {err = env.kernel_FFT_1D.setArg(0, input);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(1, sizeof(cl_int2), &inputDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(2, sizeof(cl_int2), &strideDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(3, sizeof(int), &step);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(4, sizeof(int), &N);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(5, sizeof(int), &dir);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    local = cl::NDRange(localSize[0], localSize[1]);
    global = cl::NDRange(globalSize[0], globalSize[1]);
    try{env.queue.enqueueNDRangeKernel(env.kernel_FFT_1D, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_FFT_1D" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.finish();} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    // now to do column-wise
    numGroups[0] = inputDim.s[0];
    numGroups[1] = N_windows_y;
    globalSize[0] = localSize[0]*numGroups[0];
    globalSize[1] = localSize[1]*numGroups[1];
    strideDim.s[0]=1;
    strideDim.s[1]=windowSize;
    step=inputDim.s[0];
    try {err = env.kernel_FFT_1D.setArg(2, sizeof(cl_int2), &strideDim);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_FFT_1D.setArg(3, sizeof(int), &step);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    local = cl::NDRange(localSize[0], localSize[1]);
    global = cl::NDRange(globalSize[0], globalSize[1]);
    try{env.queue.enqueueNDRangeKernel(env.kernel_FFT_1D, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_FFT_1D" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.finish();} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}

    return err;
}

