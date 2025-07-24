#ifndef GRID_FUNCTIONS_H
#define GRID_FUNCTIONS_H

void populateGrid(float* X, float* Y, cl_int2 vecDim,int windowSize, int window_shift);



void gridInterpolate(float* X_ref, float* Y_ref,float* U_ref, float* V_ref, cl_int2 vecDim_ref, float* X, float* Y,float* U, float* V, cl_int2 vecDim);

#endif
