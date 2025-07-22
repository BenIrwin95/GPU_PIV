const char* kernelSource_MaxCorr = R"(

    __kernel void findMaxCorr(__global float2* input,
                            int2 inputDim,
                            int windowSize,
                            __global float* U,
                            __global float* V,
                            int2 outputDim){

    const int lid = get_local_id(0);
    const int gid[2] = {get_group_id(0),get_group_id(1)};
    
    const int startPoint = (gid[1]*windowSize)*inputDim.x + gid[0]*windowSize;
    
    __local float colMax[256];
    __local float best_i_by_row[256];
    
    //each thread will find the max in their column
    if(lid<windowSize){
      colMax[lid]=0.0;
      for(int i=0;i<windowSize;i++){
        float val = input[startPoint + i*inputDim.x + lid].x;
        if(val > colMax[lid]){
          colMax[lid] = val;
          best_i_by_row[lid] = i;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // one thread will then find the max overall
    if(lid==0){
        float maxCorr=0.0;
        int best_i = 0;
        int best_j = 0;
        for(int j=0;j<windowSize;j++){
            if (colMax[j] > maxCorr){
                best_j = j;
                best_i = best_i_by_row[j];
                maxCorr = colMax[j];
            }
        }
        if(best_i > windowSize/2){
            best_i = best_i-windowSize;
        }
        if(best_j > windowSize/2){
            best_j = best_j - windowSize;
        }
        U[gid[1]*outputDim.x + gid[0]] = -best_j;
        V[gid[1]*outputDim.x + gid[0]] = -best_i;
    }
    
    
}
)";
