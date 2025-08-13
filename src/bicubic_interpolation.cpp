#include "standardHeader.hpp"


const std::string kernelSource_bicubic_interpolation = R"(

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


int clamp_int(int i, int min_i, int max_i){
    if(i<min_i){
        return min_i;
    } else if(i > max_i){
        return max_i;
    } else{
        return i;
    }
}


__kernel void bicubic_interpolation(__global float* x_vals, __global float* y_vals,
                                    __global float* x_ref, __global float* y_ref, int2 refDims, __global float* C,
                                    __global float* output, int2 outputDims){
    const int gid[2] = {get_global_id(0), get_global_id(1)};

    if(gid[0]<outputDims.x && gid[1]<outputDims.y){

        const float x = x_vals[gid[0]];
        const float y = y_vals[gid[1]];



        const float min_x = x_ref[0];const float max_x = x_ref[refDims.x-1];
        const float min_y = y_ref[0];const float max_y = y_ref[refDims.y-1];



        // find the ref points for where we want to interpolate
        if(x >= min_x && x<= max_x && y >= min_y && y<= max_y){
            const int dx=x_ref[1]-x_ref[0]; // grid spacing
            const int dy=y_ref[1]-y_ref[0];

            float C_local[16];
            float row_interps[4];
            int j_anchor;
            for(int j=0;j<refDims.x-1;j++){
                if(x >= x_ref[j] && x <= x_ref[j+1]){
                    j_anchor = j;
                    break;
                }
            }

            int i_anchor;
            for(int i=0;i<refDims.y-1;i++){
                if(y >= y_ref[i] && y <= y_ref[i+1]){
                    i_anchor = i;
                    break;
                }
            }

            // fill C_local padding with constant values if value requires the 4x4 grid going off the edge of ref points
            for(int i=0;i<4;i++){
                int i_ = clamp_int(-1 + i_anchor + i, 0, refDims.y-1);
                for(int j=0;j<4;j++){
                    int j_ = clamp_int(-1 + j_anchor + j, 0, refDims.x-1);
                    C_local[i*4+j] = C[i_*refDims.x + j_];
                }
            }
            float s0 = (x - x_ref[j_anchor])/dx;
            float s1 = (y - y_ref[i_anchor])/dy;
            for(int i=0;i<4;i++){
                row_interps[i] = bicubic_1D(s0, C_local[i*4], C_local[i*4+1], C_local[i*4+2], C_local[i*4+3]);
            }
            output[gid[1]*outputDims.x + gid[0]] = bicubic_1D(s1, row_interps[0], row_interps[1], row_interps[2], row_interps[3]);

        } else { // just do basic linear interpolation
            int i_anchor, j_anchor;
            // find closest row
            if(y<min_y){
                i_anchor=0;
            } else if (y>max_y){
                i_anchor = refDims.y-2; // we move back one from the edge so that we can always do the forward difference in the last section of the code
            } else {
                for(int i=0;i<refDims.y-1;i++){
                    if(y >= y_ref[i] && y<= y_ref[i+1]){
                        i_anchor = i;
                        break;
                    }
                }
            }
            // find closest column
            if(x < min_x){
                j_anchor=0;
            } else if (x > max_x){
                j_anchor = refDims.x-2;
            } else {
                for(int j=0;j<refDims.x-1;j++){
                    if(x >= x_ref[j] && x<= x_ref[j+1]){
                        j_anchor = j;
                        break;
                    }
                }
            }
            int anchor = i_anchor*refDims.x+j_anchor;
            double dCdx = (C[anchor+1] - C[anchor])/(x_ref[j_anchor+1] - x_ref[j_anchor]);
            double dCdy = (C[anchor+refDims.x] - C[anchor])/(y_ref[i_anchor+1] - y_ref[i_anchor]);
            //
            float dx = x-x_ref[j_anchor];
            float dy = y-y_ref[i_anchor];
            output[gid[1]*outputDims.x + gid[0]] = C[anchor] + dCdx*dx + dCdy*dy;
        }
    }
}



)";



alglib::spline2dinterpolant create_2D_interpolater(std::vector<double>& x, std::vector<double>& y, std::vector<double>& C){
    alglib::real_1d_array alglib_x, alglib_y, alglib_C;
    alglib_x.setcontent(x.size(), x.data());
    alglib_y.setcontent(y.size(), y.data());
    alglib_C.setcontent(C.size(), C.data());

    alglib::spline2dinterpolant s;
    alglib::spline2dbuildbicubicv(alglib_x, x.size(), alglib_y, y.size(), alglib_C, 1, s);

    return s;
}


bicubicInterp create2DSplineInterp(std::vector<double>& x, std::vector<double>& y, std::vector<double>& C){
    bicubicInterp bicubic(x,y, C);
    bicubic.min_x = x[0];
    bicubic.max_x = x[x.size()-1];
    bicubic.min_y = y[0];
    bicubic.max_y = y[y.size()-1];
    bicubic.nx = x.size();
    bicubic.ny = y.size();
    bicubic.s = create_2D_interpolater(x,y,C);
    return bicubic;
}


double interpolate_2D_bicubic(double x, double y, bicubicInterp& bicubic){
    // handles interpolation that can be both in and out of range
    if(x>= bicubic.min_x && x<=bicubic.max_x && y>= bicubic.min_y && y<=bicubic.max_y){
        // if in range then use the bicubic interpolater
        return alglib::spline2dcalc(bicubic.s, x, y);
    } else {
        // if out of range then resort to linear interpolation
        int i_anchor, j_anchor;
        // find closest row
        if(y<bicubic.min_y){
            i_anchor=0;
        } else if (y>bicubic.max_y){
            i_anchor = bicubic.ny-2; // we move back one from the edge so that we can always do the forward difference in the last section of the code
        } else {
            for(int i=0;i<bicubic.ny-1;i++){
                if(y >= bicubic.y[i] && y<= bicubic.y[i+1]){
                    i_anchor = i;
                    break;
                }
            }
        }
        // find closest column
        if(x < bicubic.min_x){
            j_anchor=0;
        } else if (x > bicubic.max_x){
            j_anchor = bicubic.nx-2;
        } else {
            for(int j=0;j<bicubic.nx-1;j++){
                if(x >= bicubic.x[j] && x<= bicubic.x[j+1]){
                    j_anchor = j;
                    break;
                }
            }
        }
        int anchor = i_anchor*bicubic.nx+j_anchor;
        double dCdx = (bicubic.C[anchor+1] - bicubic.C[anchor])/(bicubic.x[j_anchor+1] - bicubic.x[j_anchor]);
        double dCdy = (bicubic.C[anchor+bicubic.nx] - bicubic.C[anchor])/(bicubic.y[i_anchor+1] - bicubic.y[i_anchor]);
        //
        double dx = x-bicubic.x[j_anchor];
        double dy = y-bicubic.y[i_anchor];
        return bicubic.C[anchor] + dCdx*dx + dCdy*dy;
    }
}


cl_int determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, uint32_t im_width, uint32_t im_height){
    cl_int err=CL_SUCCESS;

    cl_int2 outputDims;
    outputDims.s[0] = im_width;outputDims.s[1] = im_height;

    int localSize[2] = {8,8};
    int numGroups[2] = {(int)ceil((float)outputDims.s[0]/localSize[0]), (int)ceil((float)outputDims.s[1]/localSize[1])};
    int globalSize[2] = {numGroups[0]*localSize[0], numGroups[1]*localSize[1]};
    cl::NDRange local(localSize[0], localSize[1]);
    cl::NDRange global(globalSize[0], globalSize[1]);

    // image interpolations
    // interpolating U
    try {err = env.kernel_bicubic_interpolation.setArg(0, env.x_vals_im);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(1, env.y_vals_im);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(2, env.x_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(3, env.y_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(4, sizeof(cl_int2), &piv_data.arrSize[pass-1]);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(5, env.U_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(6, env.imageShifts_x);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(7, sizeof(cl_int2), &outputDims);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_bicubic_interpolation, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_bicubic_interpolation" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();

    // interpolating V
    try {err = env.kernel_bicubic_interpolation.setArg(5, env.V_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(6, env.imageShifts_y);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_bicubic_interpolation, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_bicubic_interpolation" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();

    return err;
}


cl_int upscale_velocity_field(int pass, PIVdata& piv_data, OpenCL_env& env){
    cl_int err=CL_SUCCESS;
    env.queue.enqueueWriteBuffer( env.x_vals, CL_TRUE, 0, piv_data.x[pass].size()*sizeof(float), piv_data.x[pass].data());
    env.queue.enqueueWriteBuffer( env.y_vals, CL_TRUE, 0, piv_data.y[pass].size()*sizeof(float), piv_data.y[pass].data());

    cl_int2 outputDims = piv_data.arrSize[pass];

    int localSize[2] = {8,8};
    int numGroups[2] = {(int)ceil((float)outputDims.s[0]/localSize[0]), (int)ceil((float)outputDims.s[1]/localSize[1])};
    int globalSize[2] = {numGroups[0]*localSize[0], numGroups[1]*localSize[1]};
    cl::NDRange local(localSize[0], localSize[1]);
    cl::NDRange global(globalSize[0], globalSize[1]);

    // interpolating U
    try {err = env.kernel_bicubic_interpolation.setArg(0, env.x_vals);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(1, env.y_vals);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(2, env.x_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(3, env.y_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(4, sizeof(cl_int2), &piv_data.arrSize[pass-1]);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(5, env.U_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(6, env.U);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(7, sizeof(cl_int2), &outputDims);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_bicubic_interpolation, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_bicubic_interpolation" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();

    // interpolating V
    try {err = env.kernel_bicubic_interpolation.setArg(5, env.V_ref);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_bicubic_interpolation.setArg(6, env.V);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try{env.queue.enqueueNDRangeKernel(env.kernel_bicubic_interpolation, cl::NullRange, global, local);} catch (cl::Error& e) {std::cerr << "Error Enqueuing kernel_bicubic_interpolation" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    env.queue.finish();


    return err;
}



/*
void determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, std::vector<cl_int2>& imageShifts, uint32_t im_width, uint32_t im_height){
    std::copy( piv_data.U[pass-1].begin(), piv_data.U[pass-1].end(), piv_data.U_d[pass-1].begin() );
    std::copy( piv_data.V[pass-1].begin(), piv_data.V[pass-1].end(), piv_data.V_d[pass-1].begin() );
    bicubicInterp bicubicU =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.U_d[pass-1]);
    bicubicInterp bicubicV =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.V_d[pass-1]);
    #pragma omp parallel for
    for(uint32_t i=0;i<im_height;i++){
        for(uint32_t j=0;j<im_width;j++){
            imageShifts[i*im_width + j].s[0] = std::round(interpolate_2D_bicubic(j, i, bicubicU));
            imageShifts[i*im_width + j].s[1] = std::round(interpolate_2D_bicubic(j, i, bicubicV));
        }
    }
    #pragma omp parallel for
    for(int i=0;i<piv_data.arrSize[pass].s[1];i++){
        for(int j=0;j<piv_data.arrSize[pass].s[0];j++){
            piv_data.U[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2D_bicubic(piv_data.x_d[pass][j], piv_data.y_d[pass][i], bicubicU);
            piv_data.V[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2D_bicubic(piv_data.x_d[pass][j], piv_data.y_d[pass][i], bicubicV);
        }
    }
    env.queue.enqueueWriteBuffer( env.imageShifts, CL_TRUE, 0, imageShifts.size()*sizeof(cl_int2), imageShifts.data());
    const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
    env.queue.enqueueWriteBuffer( env.U, CL_TRUE, 0, gridSizeBytes, piv_data.U[pass].data());
    env.queue.enqueueWriteBuffer( env.V, CL_TRUE, 0, gridSizeBytes, piv_data.V[pass].data());
}*/



/*
void determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, std::vector<cl_int2>& imageShifts, uint32_t im_width, uint32_t im_height){
    std::copy( piv_data.U[pass-1].begin(), piv_data.U[pass-1].end(), piv_data.U_d[pass-1].begin() );
    std::copy( piv_data.V[pass-1].begin(), piv_data.V[pass-1].end(), piv_data.V_d[pass-1].begin() );
    bicubicInterp bicubicU =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.U_d[pass-1]);
    bicubicInterp bicubicV =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.V_d[pass-1]);
    #pragma omp parallel for
    for(uint32_t i=0;i<im_height;i++){
        for(uint32_t j=0;j<im_width;j++){
            imageShifts[i*im_width + j].s[0] = std::round(interpolate_2D_bicubic(j, i, bicubicU));
            imageShifts[i*im_width + j].s[1] = std::round(interpolate_2D_bicubic(j, i, bicubicV));
        }
    }
    #pragma omp parallel for
    for(int i=0;i<piv_data.arrSize[pass].s[1];i++){
        for(int j=0;j<piv_data.arrSize[pass].s[0];j++){
            piv_data.U[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2D_bicubic(piv_data.x_d[pass][j], piv_data.y_d[pass][i], bicubicU);
            piv_data.V[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2D_bicubic(piv_data.x_d[pass][j], piv_data.y_d[pass][i], bicubicV);
        }
    }
    env.queue.enqueueWriteBuffer( env.imageShifts, CL_TRUE, 0, imageShifts.size()*sizeof(cl_int2), imageShifts.data());
    const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
    env.queue.enqueueWriteBuffer( env.U, CL_TRUE, 0, gridSizeBytes, piv_data.U[pass].data());
    env.queue.enqueueWriteBuffer( env.V, CL_TRUE, 0, gridSizeBytes, piv_data.V[pass].data());
}*/

