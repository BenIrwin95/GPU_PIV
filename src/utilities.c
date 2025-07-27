#include "standardLibraries.h"
#include "macros.h"
#include "globalVars.h"
#include "functions.h"




void debug_message(const char* message, int debug, int min_debug, clock_t* lastTime){
  if(debug>min_debug){
    clock_t currentTime = clock();
    double time_passed = ((double) (currentTime - *lastTime)) / CLOCKS_PER_SEC;
    //*lastTime=currentTime;
    printf("%f - ", time_passed);
    for(int i=0;i<min_debug;i++){
      printf("-");
    }


    printf("%s\n",message);
  }
}



void round_float_array(float* A, int N){
  for(int i=0;i<N;i++){
    A[i] = (int)A[i];
  }
}


void printComplexArray(cl_float2* input, size_t width, size_t height){
  printf("\n");
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      int idx = i*width+j;
      printf(" (%.2f, %.2f)",input[idx].x, input[idx].y);
    }
    printf("\n");
  }
}


void multiply_double_array_by_scalar(double* arr, int arrSize, double scalar){
  for(int i=0;i<arrSize;i++){
    arr[i] = arr[i]*scalar;
  }
}


void multiply_float_array_by_scalar(float* arr, int arrSize, float scalar){
  for(int i=0;i<arrSize;i++){
    arr[i] = arr[i]*scalar;
  }
}

void initialisePIVdataMemory(PIVdata *piv_data){
  piv_data->X_passes = (float**)malloc(piv_data->N_pass * sizeof(float*));
  piv_data->Y_passes = (float**)malloc(piv_data->N_pass * sizeof(float*));
  piv_data->U_passes = (float**)malloc(piv_data->N_pass * sizeof(float*));
  piv_data->V_passes = (float**)malloc(piv_data->N_pass * sizeof(float*));
  piv_data->vecDim_passes = (cl_int2*)malloc(piv_data->N_pass * sizeof(cl_int2)); // the dimensions of the arrays in each pass
}



void freePIVdata(PIVdata *piv_data){
  for(int i=0;i<piv_data->N_pass;i++){
    free(piv_data->X_passes[i]);
    free(piv_data->Y_passes[i]);
    free(piv_data->U_passes[i]);
    free(piv_data->V_passes[i]);
  }
  free(piv_data->X_passes);
  free(piv_data->Y_passes);
  free(piv_data->U_passes);
  free(piv_data->V_passes);
  free(piv_data->vecDim_passes);
  free(piv_data->windowSizes);
}

