#include "standardHeader.hpp"


const std::string kernelSource_vectorValidation = R"(
__kernel void identifyInvalidVectors(__global float* U,
                                     __global float* V,
                                     __global int* flags,
                                     int2 vecDim){

  int gid[2] = {get_global_id(0), get_global_id(1)};

  if(gid[0]<vecDim.x && gid[1]<vecDim.y){
    int idx = gid[1]*vecDim.x + gid[0];
    float2 avg;
    avg.x=0.0;avg.y=0.0f;
    int count=0;
    for(int i=-1;i<1;i++){
      for(int j=-1;j<1;j++){
        if(i==0 && j==0){continue;}//only want neighbours
        int i_=gid[1]+i;
        int j_=gid[0]+j;
        // Check if the potential neighbor is within the array bounds
        if (i_ >= 0 && i_ < vecDim.y && j_ >= 0 && j_ < vecDim.x){
          //sum_neighbors += input_array[get_1d_index(neighbor_row, neighbor_col, cols)];
          avg.x += U[i_*vecDim.x + j_];
          avg.y += V[i_*vecDim.x + j_];
          count++;
        }
      }
    }
    avg.x=avg.x/count;avg.y=avg.y/count;
    float eps=1e-6f;
    float criterion = sqrt(pow(avg.x - U[idx],2) + pow(avg.y - V[idx],2)) + eps;
    float AVE = sqrt(pow(avg.x,2) + pow(avg.y,2));
    criterion = criterion/AVE;
    if(criterion > 0.1){
      flags[idx] = 1;
    } else {
      flags[idx] = 0;
    }
  }
}



__kernel void correctInvalidVectors(__global float* X,
                                    __global float* Y,
                                    __global float* U,
                                    __global float* V,
                                    __global int* flags,
                                    int2 vecDim){
  int gid[2] = {get_global_id(0), get_global_id(1)};
  if(gid[0]<vecDim.x && gid[1]<vecDim.y){
    int idx = gid[1]*vecDim.x + gid[0];
    if(flags[idx]==1){ // marked for replacement
      int srcDist = 4; // how far we want to look for points to interpolate from
      int i_start, i_end, j_start, j_end;
      j_start = gid[0]-srcDist;                     if(j_start<0){j_start=0;};
      j_end =gid[0]+srcDist;                        if(j_end>vecDim.x){j_end=vecDim.x;};
      i_start = gid[1]-srcDist;            if(i_start<0){i_start=0;};
      i_end = gid[1]+srcDist;              if(i_end>vecDim.y){i_end=vecDim.y;};
      // iterate through the points we will interpolate from
      float U_new=0.0f;
      float V_new=0.0f;
      float sum_weights=0.0f;
      for(int i=i_start;i<i_end;i++){
        for(int j=j_start;j<j_end;j++){
          int idx_local = i*vecDim.x+j;
          if(flags[idx_local]==0){
            float d2 = pow(X[idx]-X[idx_local],2) + pow(Y[idx]-Y[idx_local],2);
            float weight = 1.0/d2;
            sum_weights+=weight;
            U_new += weight*U[idx_local];
            V_new += weight*V[idx_local];
          }
        }
      }
      if(sum_weights>0){
        U[idx] = U_new/sum_weights;
        V[idx] = V_new/sum_weights;
      }


    }
  }
}

)";



cl_int validateVectors(int pass, PIVdata& piv_data, OpenCL_env& env){
    cl_int err = CL_SUCCESS;
    size_t localSize_1D = 8;
    size_t numGroups[2];
    numGroups[0] = ceil( (float)piv_data.arrSize[pass].s[0]/localSize_1D );
    numGroups[1] = ceil( (float)piv_data.arrSize[pass].s[1]/localSize_1D );
    cl::NDRange local(localSize_1D, localSize_1D);
    cl::NDRange global(numGroups[0]*localSize_1D, numGroups[1]*localSize_1D);
    // global size has to be a multiple of the local size, hence this somewhat convoluted setup
    // will filter the innecessary threads kernel-side


    // identify vectors that need correcting
    try {err = env.kernel_identifyInvalidVectors.setArg(0, env.U);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_identifyInvalidVectors.setArg(1, env.V);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_identifyInvalidVectors.setArg(2, env.flags);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_identifyInvalidVectors.setArg(3, sizeof(cl_int2), &piv_data.arrSize[pass]);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_identifyInvalidVectors, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_identifyInvalidVectors" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();

    // correct the bad vectors
    try {err = env.kernel_correctInvalidVectors.setArg(0, env.X);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_correctInvalidVectors.setArg(1, env.Y);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_correctInvalidVectors.setArg(2, env.U);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_correctInvalidVectors.setArg(3, env.V);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_correctInvalidVectors.setArg(4, env.flags);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_correctInvalidVectors.setArg(5, sizeof(cl_int2), &piv_data.arrSize[pass]);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_correctInvalidVectors, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_correctInvalidVectors" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();

    return err;

}


