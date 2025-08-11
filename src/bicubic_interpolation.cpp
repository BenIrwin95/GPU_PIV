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
}

