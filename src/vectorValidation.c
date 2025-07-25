#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>



int* identifyInvalidVectors(float* U, float* V, cl_int2 vecDim){
  int* flags = (int*)malloc(vecDim.x*vecDim.y*sizeof(int));
  int invalidVectors = 0;
  for(int i=1;i<vecDim.y-1;i++){
    for(int j=1;j<vecDim.x-1;j++){
      int idx = i*vecDim.x + j;
      int neighbours[8];
      neighbours[0]=idx + 1;
      neighbours[1]=idx + 1 + vecDim.x;
      neighbours[2]=idx + vecDim.x;
      neighbours[3]=idx - 1 + vecDim.x;
      neighbours[4]=idx - 1;
      neighbours[5]=idx - 1 - vecDim.x;
      neighbours[6]=idx - vecDim.x;
      neighbours[7]=idx + 1 - vecDim.x;
      cl_float2 avg;
      avg.x=0;avg.y=0;
      for( int k=0;k<8;k++){
        avg.x += U[neighbours[k]] * 0.125;
        avg.y += V[neighbours[k]] * 0.125;
      }
      float criterion = sqrt(pow(avg.x - U[idx],2) + pow(avg.y - V[idx],2));
      criterion = criterion/sqrt(pow(avg.x,2) + pow(avg.y,2));
      if(criterion > 0.2){
        flags[idx] = 1;
        invalidVectors++;
      } else {
        flags[idx] = 0;
      } 
    }
  }
  printf("%d vectors marked as invalid\n", invalidVectors);
  return flags;
}
