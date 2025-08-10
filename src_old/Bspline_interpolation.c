#include "standardLibraries.h"
#include "macros.h"
#include "globalVars.h"
#include "functions.h"



float clamp_val(float val, float min_val, float max_val){
    if(val < min_val){
        return min_val;
    } else if(val > max_val){
        return max_val;
    } else {
        return val;
    }
}

// knot vector creation (clamped)
float* create_clamped_knot_vector(int n,int degree, int* m){
    *m = n + degree + 1;
    float* t = (float*)malloc(*m*sizeof(float));
    for(int i=0;i<*m;i++){
        if(i<=degree){
            t[i]=0;
        } else if(i>degree && i < *m-degree){
            t[i]=t[i-1]+1;
        } else{
            t[i] = t[*m-degree-1];
        }
    }
    return t;
}


float* create_clamped_knot_vector_with_refValues(int n,float* refVals, int degree, int* m_out){
    int m = n + degree + 1;
    *m_out=m;
    float* t = (float*)malloc(m*sizeof(float));
    int num_internal_knots = m - 2 * degree;
    float t0 = refVals[0];
    float t1 = refVals[n - 1];

    // Fill the clamped start
    for (int i = 0; i < degree; ++i) {
        t[i] = t0;
    }

    // Fill internal knots
    for (int i = 0; i < num_internal_knots; ++i) {
        t[degree + i] = t0 + ((t1 - t0) * i) / (num_internal_knots - 1);
    }

    // Fill the clamped end
    for (int i = m - degree; i < m; ++i) {
        t[i] = t1;
    }
    return t;
}

float* create_clamped_knot_vector_between(int n,float t0,float t1, int degree, int* m_out){
    int m = n + degree + 1;
    *m_out=m;
    float* t = (float*)malloc(m*sizeof(float));
    int num_internal_knots = m - 2 * degree;
    // float t0 = refVals[0];
    // float t1 = refVals[n - 1];

    // Fill the clamped start
    for (int i = 0; i < degree; ++i) {
        t[i] = t0;
    }

    // Fill internal knots
    for (int i = 0; i < num_internal_knots; ++i) {
        t[degree + i] = t0 + ((t1 - t0) * i) / (num_internal_knots - 1);
    }

    // Fill the clamped end
    for (int i = m - degree; i < m; ++i) {
        t[i] = t1;
    }
    return t;
}



spline2D initialise_spline2D(float* X_ref, float* Y_ref, float* C, cl_int2 inputDim, cl_mem X_GPU, cl_mem Y_GPU, cl_mem C_GPU, cl_mem u_knots_GPU, cl_mem v_knots_GPU, cl_mem A, cl_mem b, cl_command_queue queue){
    spline2D out;
    out.degree=3; //hard-baked as cubic
    out.u_knots = create_clamped_knot_vector_between(inputDim.x,X_ref[0],X_ref[inputDim.x-1], out.degree, &out.mu);
    out.v_knots = create_clamped_knot_vector_between(inputDim.y,Y_ref[0],Y_ref[(inputDim.y-1)*inputDim.x], out.degree, &out.mv);
    out.X = X_ref;
    out.Y = Y_ref;
    out.C = C;
    out.nx=inputDim.x;
    out.ny=inputDim.y;
    out.X_GPU=Y_GPU;
    out.Y_GPU=X_GPU;
    out.C_GPU=C_GPU;
    out.u_knots_GPU=u_knots_GPU;
    out.v_knots_GPU=v_knots_GPU;
    // space allocated for linalg
    out.A=A;
    out.b=b;
    // copy data into the GPU
    size_t bytesize;
    bytesize = out.nx*out.ny*sizeof(float);
    clEnqueueWriteBuffer( queue, X_GPU, CL_TRUE, 0, bytesize, out.X, 0, NULL, NULL );
    clEnqueueWriteBuffer( queue, Y_GPU, CL_TRUE, 0, bytesize, out.Y, 0, NULL, NULL );
    clEnqueueWriteBuffer( queue, C_GPU, CL_TRUE, 0, bytesize, out.C, 0, NULL, NULL );
    bytesize = out.mu*sizeof(float);
    clEnqueueWriteBuffer( queue, u_knots_GPU, CL_TRUE, 0, bytesize, out.u_knots, 0, NULL, NULL );
    bytesize = out.mv*sizeof(float);
    clEnqueueWriteBuffer( queue, v_knots_GPU, CL_TRUE, 0, bytesize, out.v_knots, 0, NULL, NULL );
    //fit_spline_to_control_points(&out);
    return out;
}


void dismantle_spline2D(spline2D splineObj){
    free(splineObj.u_knots);
    free(splineObj.v_knots);
}



float Bspline_coeffs(float u, int i, int k, float* knots, int m){
    //u: the value within the knot vector we are evaluating at
    //i: the ith Bspline coefficient are evaluating
    //k: the degree of coefficient we want
    //knots: the knot vector
    //m: length of knot vector

    //handle special case at end points
    //printf("%d %d\n",m-k-1, knots[m-k-1]);
    if(u==knots[m-k-1] && i==m-k-2){
        return 1;
    }
    if(k==0){ //handle the first row of coeffs
        if( u >= knots[i] && u < knots[i + 1]){
            return 1;
        } else{
            return 0;
        }
    } else{ //use recursion to find the rest
        float denom1 = knots[i + k] - knots[i];
        float denom2 = knots[i + k + 1] - knots[i + 1];

        float term1 = 0.0f;
        float term2 = 0.0f;
        if( denom1 != 0.0f){
            term1 = (u - knots[i]) / denom1 * Bspline_coeffs(u,i, k - 1, knots, m);
        }
        if(denom2 != 0.0f){
            term2 = (knots[i + k + 1] - u) / denom2 * Bspline_coeffs(u,i + 1, k - 1, knots, m);
        }
        return term1 + term2;
    }
}





float evaluate_2D_spline_value(float u,float v, spline2D* splineObj){
    float out=0.0f;
    int nx = splineObj->nx;int ny = splineObj->ny;
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            out += splineObj->C[i*nx + j] * Bspline_coeffs(u, j, splineObj->degree, splineObj->u_knots, splineObj->mu) *  Bspline_coeffs(v, i, splineObj->degree, splineObj->v_knots, splineObj->mv);
        }
    }
    return out;
}


float evaluate_2D_spline_slope(float u, float v, int dir, spline2D* splineObj){
    if(dir==0){ // u direction
        float du = (splineObj->u_knots[splineObj->degree+1] - splineObj->u_knots[splineObj->degree])*1e-5;
        float u_plus = u+du;
        float u_neg = u-du;
        float max_u = splineObj->u_knots[splineObj->mu-1];
        float min_u = splineObj->u_knots[0];
        if(u_plus>max_u){
            u_plus = max_u;
        }
        if(u_neg < min_u){
            u_neg = min_u;
        }
        float val_plus = evaluate_2D_spline_value(u_plus, v, splineObj);
        float val_neg = evaluate_2D_spline_value(u_neg, v, splineObj);
        return (val_plus-val_neg)/(u_plus - u_neg);
    } else if(dir==1){// v direction
        float dv = (splineObj->v_knots[splineObj->degree+1] - splineObj->v_knots[splineObj->degree])*1e-5;
        float v_plus = v+dv;
        float v_neg = v-dv;
        float max_v = splineObj->v_knots[splineObj->mv-1];
        float min_v = splineObj->v_knots[0];
        if(v_plus>max_v){
            v_plus = max_v;
        }
        if(v_neg < min_v){
            v_neg = min_v;
        }
        float val_plus = evaluate_2D_spline_value(u, v_plus, splineObj);
        float val_neg = evaluate_2D_spline_value(u, v_neg, splineObj);
        return (val_plus-val_neg)/(v_plus - v_neg);
    } else {
        return 0;
    }
}




float evaluate_2D_spline(float u, float v, spline2D* splineObj){
    float max_u = splineObj->u_knots[splineObj->mu-1];
    float min_u = splineObj->u_knots[0];
    float max_v = splineObj->v_knots[splineObj->mv-1];
    float min_v = splineObj->v_knots[0];
    // is it within bounds and able to be evaluated normally
    if(u >= min_u && u<=max_u && v>=min_u && v<=max_u){
        return evaluate_2D_spline_value(u,v, splineObj);
    } else { // extrapolate linearly from edge
        // find nearest point on known grid
        float u_clamp=clamp_val(u,min_u,max_u);
        float v_clamp=clamp_val(v,min_v,max_v);


        float val = evaluate_2D_spline_value(u_clamp,v_clamp, splineObj);
        float dval_du = evaluate_2D_spline_slope(u_clamp, v_clamp, 0, splineObj);
        float dval_dv = evaluate_2D_spline_slope(u_clamp, v_clamp, 1, splineObj);
        return val + (u-u_clamp)*dval_du + (v-v_clamp)*dval_dv;
    }

}


// __global float* X_GPU,
// __global float* Y_GPU,
// __global float* C_GPU,
// __global float* A,
// __global float* b,
// int2 inputDim,
// __global float* u_knots, __local float* u_knots_local, int mu,
// __global float* v_knots, __local float* v_knots_local, int mv,
// int degree


void fit_spline_to_control_points(spline2D* splineObj, OpenCL_env *env){
    cl_int err;
    int nx = splineObj->nx;
    int ny = splineObj->ny;
    int N = nx*ny;

    size_t localSize[2]={5,5};

    size_t numGroups[2] = {ceil( (float)N/localSize[0] ), ceil( (float)N/localSize[1] )};
    size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};
    cl_int2 inputDim;
    inputDim.x=splineObj->nx;inputDim.y=splineObj->ny;
    int idx=0;
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->X_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->Y_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->C_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->A); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->b); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_int2), &inputDim); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->u_knots_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(float) * splineObj->mu, NULL); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(int), &splineObj->mu); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(cl_mem), &splineObj->v_knots_GPU); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(float) * splineObj->mv, NULL); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(int), &splineObj->mv); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clSetKernelArg(env->kernel_populate_spline_fitting_matrix, idx, sizeof(int), &splineObj->degree); idx++;if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clEnqueueNDRangeKernel(env->queue, env->kernel_populate_spline_fitting_matrix, 2, NULL, globalSize, localSize,0, NULL, NULL);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}
    err=clFinish(env->queue);if(err!=CL_SUCCESS){ERROR_MSG_OPENCL(err);}

    // printf("making matrix A\n");
    // float* Bu = (float*)malloc(nx*sizeof(float));
    // float* Bv = (float*)malloc(ny*sizeof(float));
    //
    // // construct matrix to solve for coeffs
    // float* A = (float*)malloc(N*N*sizeof(float));
    // for(int i=0;i<ny;i++){
    //     for(int j=0;j<nx;j++){
    //         int rowIdx = i*nx+j;
    //         float x_ref = splineObj->X[rowIdx];
    //         float y_ref = splineObj->Y[rowIdx];
    //         //precompute coeffs
    //         for(int ii=0;ii<ny;ii++){
    //             Bv[ii] = Bspline_coeffs(y_ref, ii, splineObj->degree, splineObj->v_knots, splineObj->mv);
    //         }
    //         for(int jj=0;jj<nx;jj++){
    //             Bu[jj] = Bspline_coeffs(x_ref, jj, splineObj->degree, splineObj->u_knots, splineObj->mu);
    //         }
    //     }
    // }
    // free(Bu);free(Bv);
    //
    //
    // printf("done\n");
    // free(A);
}




extern const char* kernelSource_Bsplines = R"(

// recursion is not allowed with openCL and thus this is needed instead
float calculate_basis_function(float u, int i, int p, __local float* knots, int m) {
    // p: degree
    // i: control point index
    // u: parameter value

    //handle special case at end point
    if(u==knots[m-p-1] && i==m-p-2){
        return 1.0;
    }
    const int p_max = 5; //hardcoded
    float B[p_max + 1];

    // Base case: degree 0
    // N_j,0(t) is 1 if t is in [knot_j, knot_{j+1}) and 0 otherwise.
    for (int j = 0; j <= p; ++j) {
        if (u >= knots[i + j] && u < knots[i + j + 1]) {
            B[j] = 1.0f;
        } else {
            B[j] = 0.0f;
        }
    }

    // Iterative step: from degree 1 up to p
    for (int k = 1; k <= p; ++k) {
        for (int j = 0; j <= p - k; ++j) {
            float denom1 = knots[i + j + k] - knots[i + j];
            float denom2 = knots[i + j + k + 1] - knots[i + j + 1];

            float term1 = 0.0f;
            if (denom1 > 1e-6) { // Use a small epsilon to avoid division by zero
                term1 = (u - knots[i + j]) / denom1 * B[j];
            }

            float term2 = 0.0f;
            if (denom2 > 1e-6) { // Use a small epsilon
                term2 = (knots[i + j + k + 1] - u) / denom2 * B[j + 1];
            }

            B[j] = term1 + term2;
        }
    }
    return B[0]; // The result is the first element of the final array
}




__kernel void populate_spline_fitting_matrix(__global float* X_GPU,
                                             __global float* Y_GPU,
                                             __global float* C_GPU,
                                             __global float* A,
                                             __global float* b,
                                             int2 inputDim,
                                             __global float* u_knots, __local float* u_knots_local, int mu,
                                             __global float* v_knots, __local float* v_knots_local, int mv,
                                             int degree){

    const int gid[2] = {get_global_id(0), get_global_id(1)};
    const int lid[2] = {get_local_id(0), get_local_id(1)};
    const int N = inputDim.x*inputDim.y;
    if(gid[0]<N && gid[1]<N){ // due to how GPUs work, sometimes you have more threads than necessary
        // load the knot vectors locally to avoid repeated memory access
        // no guarantee of having enough local threads so we will do this manually
        if(lid[0]==0 && lid[1]==0){
            for(int i=0;i<mu;i++){
                u_knots_local[i] = u_knots[i];
            }
            for(int i=0;i<mv;i++){
                v_knots_local[i] = v_knots[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // figure out the coords of the point being influence
        const float x_ref = X_GPU[gid[0]];
        const float y_ref = Y_GPU[gid[0]];

        // figure out the location of the points doing the influencing
        int ii = floor((float)gid[1]/inputDim.x);
        int jj = gid[1] - ii*inputDim.x;

        float Bv = calculate_basis_function(y_ref, ii, degree, v_knots_local, mv);
        float Bu = calculate_basis_function(x_ref, jj, degree, u_knots_local, mu);


        A[gid[0]*N + gid[1]] = Bu*Bv;
        b[gid[0]] = C_GPU[gid[0]];
    }
}
)";


