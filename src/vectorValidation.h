#ifndef VECTOR_VALIDATION_H
#define VECTOR_VALIDATION_H

int* identifyInvalidVectors(float* U, float* V, cl_int2 vecDim);


void correctInvalidVectors(float* X,float* Y, float* U, float* V, cl_int2 vecDim, int* flags);

void validateVectors(float* X,float* Y, float* U, float* V, cl_int2 vecDim);

#endif
