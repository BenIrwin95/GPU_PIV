// standard headers
#include "standardLibraries.h"
#include "macros.h"
#include "globalVars.h"
#include "functions.h"



int main(){

    cl_int err;
    OpenCL_env env;
    err= initialise_OpenCL(&env);
    if(err!=CL_SUCCESS){
        return 1;
    }


    int nx=4;
    int ny=4;
    int N=nx*ny;
    cl_int2 inputDim;
    inputDim.x=nx;inputDim.y=ny;
    float* X = (float*)malloc(nx*ny*sizeof(float));
    float* Y = (float*)malloc(nx*ny*sizeof(float));
    float* C = (float*)malloc(nx*ny*sizeof(float));

    float* x = (float*)malloc(nx*sizeof(float));
    float* y = (float*)malloc(ny*sizeof(float));

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
            printf(" %.2f", C[i*nx + j]);
        }printf("\n");
    } printf("\n");

    // initialise some space on GPU
    cl_mem X_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
    cl_mem Y_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
    cl_mem C_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
    cl_mem A_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, N*N*sizeof(float), NULL, &err);
    cl_mem b_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
    int mu = nx + 3 + 1;
    cl_mem u_knots_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, mu*sizeof(float), NULL, &err);
    int mv = ny + 3 + 1;
    cl_mem v_knots_GPU = clCreateBuffer(env.context, CL_MEM_READ_WRITE, mv*sizeof(float), NULL, &err);

    // initialise spline
    spline2D splineObj = initialise_spline2D(X, Y, C, inputDim, X_GPU, Y_GPU, C_GPU, u_knots_GPU, v_knots_GPU, A_GPU, b_GPU, env.queue);
    printf("fitting...\n");
    fit_spline_to_control_points(&splineObj, &env);
    printf("done\n");

    float* A = (float*)malloc(N*N*sizeof(float));
    clEnqueueReadBuffer(env.queue, A_GPU, CL_TRUE, 0, N*N*sizeof(float), A, 0, NULL, NULL );
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf(" %.2f", A[i*N + j]);
        }printf("\n");
    } printf("\n");
    free(A);
//
//
//
//     for(int i=0;i<ny;i++){
//         for(int j=0;j<nx;j++){
//             printf(" %.2f", C[i*nx + j]);
//         }printf("\n");
//     } printf("\n");
//
//
//     // make some points to interpolate at
//     int n_interp = 7;
//     float* x_interp = (float*)malloc(n_interp*sizeof(float));
//     float* y_interp = (float*)malloc(n_interp*sizeof(float));
//     float* z = (float*)malloc(n_interp*n_interp*sizeof(float));
//
//     for(int j=0;j<n_interp;j++){
//         x_interp[j]=x[0] + j*(x[nx-1]-x[0])/(n_interp-1);
//     }
//     for(int i=0;i<n_interp;i++){
//         y_interp[i]=y[0] + i*(y[ny-1]-y[0])/(n_interp-1);
//     }
//
//     for(int i=0;i<n_interp;i++){
//         for(int j=0;j<n_interp;j++){
//             //printf(" (%.2f %.2f)",x_interp[j],y_interp[i]);
//             z[i*n_interp+j] = evaluate_2D_spline(x_interp[j],y_interp[i], &splineObj);
//             printf(" %.2f",z[i*n_interp+j]);
//         }printf("\n");
//     } printf("\n");
//
    dismantle_spline2D(splineObj);
    clReleaseMemObject(X_GPU);clReleaseMemObject(Y_GPU);clReleaseMemObject(C_GPU);clReleaseMemObject(u_knots_GPU);clReleaseMemObject(v_knots_GPU);clReleaseMemObject(A_GPU);clReleaseMemObject(b_GPU);
    free(X);free(Y);
    free(x);free(y);free(C);
    //free(x_interp);free(y_interp);free(z);
    close_OpenCL(&env);
    return 0;
}
