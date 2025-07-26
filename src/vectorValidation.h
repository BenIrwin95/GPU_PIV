#ifndef VECTOR_VALIDATION_H
#define VECTOR_VALIDATION_H

extern const char* kernelSource_vectorValidation;

void validateVectors(cl_mem X, cl_mem Y, cl_mem U, cl_mem V, cl_mem flags, cl_int2 vecDim, cl_kernel kernel_identifyInvalidVectors, cl_kernel kernel_correctInvalidVectors, cl_command_queue queue);

#endif
