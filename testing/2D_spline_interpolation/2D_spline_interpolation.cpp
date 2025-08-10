#include "standardHeader.hpp"



void displayArray(std::vector<float>& A, int nx, int ny){
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            std::cout << " "<< std::setprecision(2) << A[i*nx+j];
        }
        std::cout << std::endl;
    }
}


int main(){

    cl_int err;
    OpenCL_env env;
    if(env.status != CL_SUCCESS){CHECK_CL_ERROR(env.status);return 1;}

    int nx=6;
    int ny=6;
    cl_int2 arrSize;
    arrSize.s[0]=nx;arrSize.s[1]=ny;
    int N=nx*ny;
    std::vector<float> X(N);
    std::vector<float> x(nx);
    std::vector<float> Y(N);
    std::vector<float> y(ny);
    std::vector<float> C(N);


    for(int j=0;j<nx;j++){
        x[j]=j;
    }
    for(int i=0;i<ny;i++){
        y[i]=i;
    }

    // populate data
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            //printf(" (%.2f %.2f)",x[j],y[i]);
            C[i*nx + j] = x[j]*x[j] + y[i]*y[i];
            X[i*nx + j] = x[j];
            Y[i*nx + j] = y[i];
        }
    }

    displayArray(C, nx, ny);



    std::vector<double> x_double(x.begin(), x.end());
    std::vector<double> y_double(y.begin(), y.end());
    //std::vector<double> C_double(C.begin(), C.end());
    std::vector<double> C_double(N);


    std::copy( C.begin(), C.end(), C_double.begin() );

    //

    splineInterp splineObj =  create2DSplineInterp(x_double, y_double, C_double);
    //double interpolate_2Dspline(double x, double y, splineInterp& spline);

    // alglib::real_1d_array alglib_x, alglib_y, alglib_C;
    // alglib_x.setcontent(x_double.size(), x_double.data());
    // alglib_y.setcontent(y_double.size(), y_double.data());
    // alglib_C.setcontent(C_double.size(), C_double.data());
    //
    // alglib::spline2dinterpolant s;
    // alglib::spline2dbuildbicubicv(alglib_x, x_double.size(), alglib_y, y_double.size(), alglib_C, 1, s);
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            //printf(" (%.2f %.2f)",x[j],y[i]);
            //C[i*nx + j] = alglib::spline2dcalc(s, x_double[j]-1, y_double[i]-1);;
            C[i*nx + j] = interpolate_2Dspline(x_double[j]-1, y_double[i]-1, splineObj);
        }
    }
    displayArray(C, nx, ny);






    // spline_OpenCL spline;
    // spline.degree=2;
    // spline.u_knots = cl::Buffer(env.context, CL_MEM_READ_WRITE, (nx+spline.degree+1)*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return 1;}
    // spline.v_knots = cl::Buffer(env.context, CL_MEM_READ_WRITE, (ny+spline.degree+1)*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return 1;}
    // std::vector<float> u_knots(nx+spline.degree+1);
    // std::vector<float> v_knots(ny+spline.degree+1);
    //
    // initialise_knot_vectors(spline, X, Y, arrSize, env);
    //
    // env.queue.enqueueReadBuffer(spline.u_knots, CL_TRUE, 0, spline.mu*sizeof(float), u_knots.data());
    // env.queue.enqueueReadBuffer(spline.v_knots, CL_TRUE, 0, spline.mu*sizeof(float), v_knots.data());
    //
    //
    // displayArray(u_knots, spline.mu, 1);
    // displayArray(v_knots, spline.mv, 1);

    return 0;
}
