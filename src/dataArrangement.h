#ifndef DATA_ARRANGEMENT_H
#define DATA_ARRANGEMENT_H

extern const char* kernelSource_uniformTiling;

void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int windowShift, cl_int2 vecDim, cl_mem output, cl_kernel kernel_uniformTiling, cl_command_queue queue);

void offset_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int window_shift, cl_mem U_GPU, cl_mem V_GPU, cl_int2 vecDim, cl_mem output, cl_kernel kernel_offsetTiling, cl_command_queue queue);

#endif
