#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>


// Macro to find the maximum of two values
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Macro to find the minimum of two values
#define MIN(a, b) ((a) < (b) ? (a) : (b))



float calcDist2(float X1, float Y1, float X2, float Y2){
  return pow(X1-X2,2) + pow(Y1-Y2,2);
}


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
      if(criterion > 0.1){
        flags[idx] = 1;
        invalidVectors++;
      } else {
        flags[idx] = 0;
      } 
    }
  }
  //printf("%d vectors marked as invalid\n", invalidVectors);
  return flags;
}



// works but is very slow
void correctInvalidVectors(float* X,float* Y, float* U, float* V, cl_int2 vecDim, int* flags){
  for(int k=0;k<vecDim.x*vecDim.y;k++){
    if(flags[k]==1){ // find a point whose value needs interpolating
      // define search area to interpolate from
      int k_row = k / vecDim.x;
      int k_col = k % vecDim.x;

      int i_start, i_end, j_start, j_end;
      int srcDist = 4; // how far we want to look for points to interpolate from
      j_start = MAX(0, k_col-srcDist);
      j_end =MIN(vecDim.x,k_col+srcDist);
      i_start = MAX(0, k_row-srcDist*vecDim.x);
      i_end = MIN(vecDim.y, k_row+srcDist*vecDim.x);

      // iterate through the points we will interpolate from
      float U_new=0.0;
      float V_new=0.0;
      float sum_weights=0.0;
      for(int i=i_start;i<i_end;i++){
        for(int j=j_start;j<j_end;j++){
          int idx = i*vecDim.x+j;
          if(flags[idx]==0){
            float d2 = calcDist2(X[k], Y[k], X[idx], Y[idx]);
            float weight = 1.0/d2;
            sum_weights+=weight;
            U_new += weight*U[idx];
            V_new += weight*V[idx];
          }
        }
      }
      U[k] = U_new/sum_weights;
      V[k] = V_new/sum_weights;
    }
  }
}



void validateVectors(float* X,float* Y, float* U, float* V, cl_int2 vecDim){
  // identify vectors that need correcting
  int* flags = identifyInvalidVectors(U, V, vecDim);
  correctInvalidVectors( X, Y,  U, V, vecDim, flags);
  free(flags);
}
