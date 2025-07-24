#ifndef DATA_ARRANGEMENT_H
#define DATA_ARRANGEMENT_H

extern const char* kernelSource_uniformTiling;

void uniformly_tile_data(cl_mem input, cl_int2 inputDim, int windowSize, int windowShift, cl_int2 vecDim, cl_mem output, cl_kernel kernel_uniformTiling, cl_command_queue queue);

#endif
