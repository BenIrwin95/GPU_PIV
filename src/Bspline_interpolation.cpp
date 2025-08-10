#include "standardHeader.hpp"


alglib::spline2dinterpolant create_2D_interpolater(std::vector<double>& x, std::vector<double>& y, std::vector<double>& C){
    alglib::real_1d_array alglib_x, alglib_y, alglib_C;
    alglib_x.setcontent(x.size(), x.data());
    alglib_y.setcontent(y.size(), y.data());
    alglib_C.setcontent(C.size(), C.data());

    alglib::spline2dinterpolant s;
    alglib::spline2dbuildbicubicv(alglib_x, x.size(), alglib_y, y.size(), alglib_C, 1, s);

    return s;
}


splineInterp create2DSplineInterp(std::vector<double>& x, std::vector<double>& y, std::vector<double>& C){
    splineInterp spline(x,y, C);
    spline.min_x = x[0];
    spline.max_x = x[x.size()-1];
    spline.min_y = y[0];
    spline.max_y = y[y.size()-1];
    spline.nx = x.size();
    spline.ny = y.size();
    spline.s = create_2D_interpolater(x,y,C);
    return spline;
}


double interpolate_2Dspline(double x, double y, splineInterp& spline){
    // handles interpolation that can be both in and out of range
    if(x>= spline.min_x && x<=spline.max_x && y>= spline.min_y && y<=spline.max_y){
        // if in range then use the spline interpolater
        return alglib::spline2dcalc(spline.s, x, y);
    } else {
        // if out of range then resort to linear interpolation
        int i_anchor, j_anchor;
        // find closest row
        if(y<spline.min_y){
            i_anchor=0;
        } else if (y>spline.max_y){
            i_anchor = spline.ny-2; // we move back one from the edge so that we can always do the forward difference in the last section of the code
        } else {
            for(int i=0;i<spline.ny-1;i++){
                if(y >= spline.y[i] && y<= spline.y[i+1]){
                    i_anchor = i;
                    break;
                }
            }
        }
        // find closest column
        if(x < spline.min_x){
            j_anchor=0;
        } else if (x > spline.max_x){
            j_anchor = spline.nx-2;
        } else {
            for(int j=0;j<spline.nx-1;j++){
                if(x >= spline.x[j] && x<= spline.x[j+1]){
                    j_anchor = j;
                    break;
                }
            }
        }
        int anchor = i_anchor*spline.nx+j_anchor;
        double dCdx = (spline.C[anchor+1] - spline.C[anchor])/(spline.x[j_anchor+1] - spline.x[j_anchor]);
        double dCdy = (spline.C[anchor+spline.nx] - spline.C[anchor])/(spline.y[i_anchor+1] - spline.y[i_anchor]);
        //
        double dx = x-spline.x[j_anchor];
        double dy = y-spline.y[i_anchor];
        return spline.C[anchor] + dCdx*dx + dCdy*dy;
    }
}



void determine_image_shifts(int pass, PIVdata& piv_data, OpenCL_env& env, std::vector<cl_int2>& imageShifts, uint32_t im_width, uint32_t im_height){
    std::copy( piv_data.U[pass-1].begin(), piv_data.U[pass-1].end(), piv_data.U_d[pass-1].begin() );
    std::copy( piv_data.V[pass-1].begin(), piv_data.V[pass-1].end(), piv_data.V_d[pass-1].begin() );
    splineInterp splineU =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.U_d[pass-1]);
    splineInterp splineV =  create2DSplineInterp(piv_data.x_d[pass-1], piv_data.y_d[pass-1], piv_data.V_d[pass-1]);
    #pragma omp parallel for
    for(uint32_t i=0;i<im_height;i++){
        for(uint32_t j=0;j<im_width;j++){
            imageShifts[i*im_width + j].s[0] = std::round(interpolate_2Dspline(j, i, splineU));
            imageShifts[i*im_width + j].s[1] = std::round(interpolate_2Dspline(j, i, splineV));
        }
    }
    #pragma omp parallel for
    for(int i=0;i<piv_data.arrSize[pass].s[1];i++){
        for(int j=0;j<piv_data.arrSize[pass].s[0];j++){
            piv_data.U[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2Dspline(piv_data.x_d[pass][j], piv_data.y_d[pass][i], splineU);
            piv_data.V[pass][i*piv_data.arrSize[pass].s[0] + j] = interpolate_2Dspline(piv_data.x_d[pass][j], piv_data.y_d[pass][i], splineV);
        }
    }
    env.queue.enqueueWriteBuffer( env.imageShifts, CL_TRUE, 0, imageShifts.size()*sizeof(cl_int2), imageShifts.data());
    const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
    env.queue.enqueueWriteBuffer( env.U, CL_TRUE, 0, gridSizeBytes, piv_data.U[pass].data());
    env.queue.enqueueWriteBuffer( env.V, CL_TRUE, 0, gridSizeBytes, piv_data.V[pass].data());
}

