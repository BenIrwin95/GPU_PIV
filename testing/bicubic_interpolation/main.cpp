#include "standardHeader.hpp"


float bicubic_1D(float s, float p_n1, float p0, float p1, float p2){
    //https://en.wikipedia.org/wiki/Bicubic_interpolation
    // bicubibc convolution algorithm for a=-0.5;
    float s2=s*s;
    float s3=s2*s;
    float term0 = 1.0f * 2.0f*p0;
    float term1 = s * (-p_n1 + p1);
    float term2 = s2 * (2.0f*p_n1 - 5.0f*p0 + 4.0f*p1 - p2);
    float term3 = s3 * (-p_n1 + 3.0f*p0 - 3.0f*p1 + p2);
    return 0.5f * (term0 + term1 + term2 + term3);
}

float bicubic_2D(float s0, float s1, std::vector<float>& C){
    std::vector<float> row_interps(4);
    for(int i=0;i<4;i++){
        row_interps[i] = bicubic_1D(s0, C[i*4], C[i*4+1], C[i*4+2], C[i*4+3]);
    }
    return bicubic_1D(s1, row_interps[0],row_interps[1], row_interps[2],row_interps[3]);
}


std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> linspaced;

    if (num <= 0) {
        return linspaced; // Return an empty vector for non-positive numbers
    }

    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    float step = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        linspaced.push_back(start + step * i);
    }
    return linspaced;
}


int main(){
    // original test on cpu
    std::vector<float> x(4);
    std::vector<float> y(4);
    std::vector<float> C(16);

    for(int i=0;i<4;i++){
        y[i]=i;
        for(int j=0;j<4;j++){
            if(i==0){x[j]=j;}
            C[i*4+j] = x[j]*x[j] + 2*y[i];
            std::cout << " " << C[i*4+j];
        }
        std::cout << std::endl;
    }
    float dx = x[1]-x[0];
    float dy = y[1]-y[0];
    int N_interp = 10;
    // std::vector<float> x_interp = linspace(x[1], x[2], N_interp);
    // std::vector<float> y_interp = linspace(y[1], y[2], N_interp);
    // std::vector<float> C_interp(N_interp*N_interp);

    // for(int i=0;i<N_interp;i++){
    //     for(int j=0;j<N_interp;j++){
    //         float s0 = (x_interp[j] - x[1])/dx;
    //         float s1 = (y_interp[i] - y[1])/dy;
    //         C_interp[i*N_interp+j] = bicubic_2D(s0,s1,C);
    //         std::cout << " " << C_interp[i*N_interp+j];
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<float> x_interp = linspace(x[0], x[3], N_interp);
    std::vector<float> y_interp = linspace(y[0], y[3], N_interp);
    std::vector<float> C_interp(N_interp*N_interp);
    for(int i=0;i<N_interp;i++){
        x_interp[i]-=1;
        y_interp[i]-=1;
    }

    // with GPU
    cl_int err;
    OpenCL_env env;
    if(env.status != CL_SUCCESS){CHECK_CL_ERROR(env.status);return 1;}
    std::cout << "making buffers" << std::endl;
    cl::Buffer x_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, x.size()*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    cl::Buffer y_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, y.size()*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    cl::Buffer C_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, C.size()*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    cl::Buffer x_vals = cl::Buffer(env.context, CL_MEM_READ_WRITE, x_interp.size()*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    cl::Buffer y_vals = cl::Buffer(env.context, CL_MEM_READ_WRITE, y_interp.size()*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    cl::Buffer output = cl::Buffer(env.context, CL_MEM_READ_WRITE, N_interp*N_interp*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    std::cout << "filling buffers" << std::endl;
    env.queue.enqueueWriteBuffer( x_ref, CL_TRUE, 0, x.size()*sizeof(float), x.data());
    env.queue.enqueueWriteBuffer( y_ref, CL_TRUE, 0, y.size()*sizeof(float), y.data());
    env.queue.enqueueWriteBuffer( C_ref, CL_TRUE, 0, C.size()*sizeof(float), C.data());
    env.queue.enqueueWriteBuffer( x_vals, CL_TRUE, 0, x_interp.size()*sizeof(float), x_interp.data());
    env.queue.enqueueWriteBuffer( y_vals, CL_TRUE, 0, y_interp.size()*sizeof(float), y_interp.data());

    cl_int2 refDims;refDims.s[0]=4;refDims.s[1]=4;
    cl_int2 outputDims;outputDims.s[0]=N_interp;outputDims.s[1]=N_interp;

    std::cout << "setting arguments" << std::endl;
    int localSize[2] = {8,8};
    int numGroups[2] = {(int)ceil((float)outputDims.s[0]/localSize[0]), (int)ceil((float)outputDims.s[1]/localSize[1])};
    int globalSize[2] = {numGroups[0]*localSize[0], numGroups[1]*localSize[1]};
    try {err = env.kernel_bicubic_interpolation.setArg(0, x_vals);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(1, y_vals);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(2, x_ref);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(3, y_ref);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(4, sizeof(cl_int2), &refDims);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 4" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(5, C_ref);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 3" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(6, output);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 5" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    try {err = env.kernel_bicubic_interpolation.setArg(7, sizeof(cl_int2), &outputDims);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 6" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    cl::NDRange local(localSize[0], localSize[1]);
    cl::NDRange global(globalSize[0], globalSize[1]);
    std::cout << "running kernel" << std::endl;
    try{env.queue.enqueueNDRangeKernel(env.kernel_bicubic_interpolation, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_bicubic_interpolation" << std::endl;CHECK_CL_ERROR(e.err());return 1;}
    env.queue.finish();

    env.queue.enqueueReadBuffer(output, CL_TRUE, 0, N_interp*N_interp*sizeof(float), C_interp.data());
    for(int i=0;i<N_interp;i++){
        for(int j=0;j<N_interp;j++){
            std::cout << " " << C_interp[i*N_interp+j];
        }
        std::cout << std::endl;
    }

    // __kernel void bicubic_interpolation(__global float* x_vals, __global float* y_vals,
    //                                     __global float* x_ref, __global float* y_ref, int2 refDims, __global float* C,
    //                                     __global float* output, int2 outputDims){



    return 0;
}
