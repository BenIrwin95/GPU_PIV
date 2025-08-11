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
    std::vector<float> x_interp = linspace(x[1], x[2], N_interp);
    std::vector<float> y_interp = linspace(y[1], y[2], N_interp);
    std::vector<float> C_interp(N_interp*N_interp);

    for(int i=0;i<N_interp;i++){
        for(int j=0;j<N_interp;j++){
            float s0 = (x_interp[j] - x[1])/dx;
            float s1 = (y_interp[i] - y[1])/dy;
            C_interp[i*N_interp+j] = bicubic_2D(s0,s1,C);
            std::cout << " " << C_interp[i*N_interp+j];
        }
        std::cout << std::endl;
    }

    return 0;
}
