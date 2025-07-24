#include <CL/cl.h>


void populateGrid(float* X, float* Y, cl_int2 vecDim,int windowSize, int window_shift){
    for(int i=0;i<vecDim.y;i++){
        for(int j=0;j<vecDim.x;j++){
            X[i*vecDim.x + j] = (float)windowSize/2 + j*window_shift;
            Y[i*vecDim.x + j] = (float)windowSize/2 + i*window_shift;
        }
    }

}


void gridInterpolate(float* X_ref, float* Y_ref,float* U_ref, float* V_ref, cl_int2 vecDim_ref, float* X, float* Y,float* U, float* V, cl_int2 vecDim){
    for(int idx=0;idx<vecDim.x*vecDim.y;idx++){
        float x = X[idx];
        float y = Y[idx];
        // find bounding x values
        int j_anchor;
        for(int j=0;j<vecDim_ref.x-1;j++){
            if(x > X_ref[j] && x < X_ref[j+1]){
                j_anchor=j;
                break;
            }
            if(j==0 && x<X_ref[j]){
                j_anchor=0;
                break;
            }
            if(j==vecDim_ref.x-1 && x>X_ref[j+1]){
                j_anchor=vecDim_ref.x-1;
                break;
            }
        }
        // find bounding y values
        int i_anchor;
        for(int i=0;i<vecDim_ref.y-1;i++){
            if(y > Y_ref[i*vecDim_ref.x] && y < Y_ref[(i+1)*vecDim_ref.x]){
                i_anchor=i;
                break;
            }
            if(i==0 && y<Y_ref[i*vecDim_ref.x]){
                i_anchor=0;
                break;
            }
            if(i==vecDim_ref.y-1 && x>X_ref[(i+1)*vecDim_ref.x]){
                i_anchor=vecDim_ref.y-1;
                break;
            }
        }

        // standard interpolation

        if(i_anchor>=0 && j_anchor >=0){
            int anchor = i_anchor*vecDim_ref.x+j_anchor;
            float dUdx = (U_ref[anchor+1]-U_ref[anchor])/(X_ref[anchor+1]-X_ref[anchor]);
            float dVdx = (V_ref[anchor+1]-V_ref[anchor])/(X_ref[anchor+1]-X_ref[anchor]);
            float dUdy = (U_ref[anchor+vecDim_ref.x]-U_ref[anchor])/(Y_ref[anchor+vecDim_ref.x]-Y_ref[anchor]);
            float dVdy = (V_ref[anchor+vecDim_ref.x]-V_ref[anchor])/(Y_ref[anchor+vecDim_ref.x]-Y_ref[anchor]);

            float dx = x-X_ref[anchor];float dy = y-Y_ref[anchor];
            U[idx] = U_ref[anchor] + dUdx*dx + dUdy*dy;
            V[idx] = V_ref[anchor] + dVdx*dx + dVdy*dy;
        }
    }
}
