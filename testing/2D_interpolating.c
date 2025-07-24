#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "gridFunctions.h"


void printArray(float* input, cl_int2 dim){
    for(int i=0;i<dim.y;i++){
        for(int j=0;j<dim.x;j++){
            int idx=i*dim.x + j;
            printf(" %.2f", input[idx]);
        }
        printf("\n");
    }
}



int main(){


    // create some test data
    cl_int2 vecDim_ref;
    vecDim_ref.x=4;
    vecDim_ref.y=4;
    float A_ref[vecDim_ref.x*vecDim_ref.y];
    float X_ref[vecDim_ref.x*vecDim_ref.y];
    float Y_ref[vecDim_ref.x*vecDim_ref.y];
    for(int i=0;i<vecDim_ref.y;i++){
        for(int j=0;j<vecDim_ref.x;j++){
            int idx=i*vecDim_ref.x + j;
            A_ref[idx] = i+j;
            X_ref[idx] = j;
            Y_ref[idx] = i;
        }
    }
    printf("-------------------\nX_ref\n");
    printArray(X_ref, vecDim_ref);
    printf("-------------------\nY_ref\n");
    printArray(Y_ref, vecDim_ref);
    printf("-------------------\nA_ref\n");
    printArray(A_ref, vecDim_ref);

    // create finer grid
    cl_int2 vecDim;
    vecDim.x=8;
    vecDim.y=8;
    float A[vecDim.x*vecDim.y];
    float X[vecDim.x*vecDim.y];
    float Y[vecDim.x*vecDim.y];
    for(int i=0;i<vecDim.y;i++){
        for(int j=0;j<vecDim.x;j++){
            int idx=i*vecDim.x + j;
            X[idx] = j*0.5 -1;
            Y[idx] = i*0.5 -1;
        }
    }
    gridInterpolate(X_ref, Y_ref,A_ref, A_ref, vecDim_ref, X, Y,A, A, vecDim);
    printf("-------------------\nX\n");
    printArray(X, vecDim);
    printf("-------------------\nY\n");
    printArray(Y, vecDim);
    printf("-------------------\nA\n");
    printArray(A, vecDim);


    return 0;
}
