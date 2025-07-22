extern const char* kernelSource_FFT_functions;


const char* getOpenCLErrorString(cl_int err);


void printComplexArray(cl_float2* input, size_t width, size_t height);


void FFT2D_tiled (cl_mem inputGPU, cl_int2 inputDim, int windowSize, int dir, cl_kernel kernelFFT_1D, cl_command_queue queue);

void FFT_corr_tiled (cl_mem input1,cl_mem input2, cl_int2 inputDim, int windowSize, cl_kernel kernelFFT_1D, cl_kernel kernelMultiplyConj, cl_command_queue queue);
