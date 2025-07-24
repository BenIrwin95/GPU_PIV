extern const char* kernelSource_complex_multiply_conjugate_norm;
extern const char* kernelSource_MaxCorr;


void FFT_corr_tiled (cl_mem input1,cl_mem input2, cl_int2 inputDim, int windowSize, cl_kernel kernelFFT_1D, cl_kernel kernelMultiplyConj, cl_command_queue queue);
