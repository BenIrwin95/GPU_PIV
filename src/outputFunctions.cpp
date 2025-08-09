#include "standardHeader.hpp"



void add_pass_data_to_file(int pass, std::ofstream& outputFile, PIVdata& piv_data, OpenCL_env& env){
    // first pull the necessary data out of the GPU
    const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
    env.queue.enqueueReadBuffer(env.X, CL_TRUE, 0, gridSizeBytes, piv_data.X[pass].data());
    env.queue.enqueueReadBuffer(env.Y, CL_TRUE, 0, gridSizeBytes, piv_data.Y[pass].data());
    env.queue.enqueueReadBuffer(env.U, CL_TRUE, 0, gridSizeBytes, piv_data.U[pass].data());
    env.queue.enqueueReadBuffer(env.V, CL_TRUE, 0, gridSizeBytes, piv_data.V[pass].data());


    outputFile << "Pass " << pass + 1 << " of " << piv_data.N_pass << "\n";
    outputFile << "Window size " << piv_data.window_sizes[pass] << "\n";
    outputFile << "Rows " << piv_data.arrSize[pass].s[1] << "\n";
    outputFile << "Cols " << piv_data.arrSize[pass].s[0] << "\n";
    outputFile << "image_x,image_y,U,V\n";
    outputFile << std::fixed << std::setprecision(2); // Set precision for floats X and Y
    for (int i = 0; i < piv_data.arrSize[pass].s[1]; i++) {
        for (int j = 0; j < piv_data.arrSize[pass].s[0]; j++) {
            int index = i * piv_data.arrSize[pass].s[0] + j;
            outputFile << piv_data.X[pass][index] << "," << piv_data.Y[pass][index] << ",";
            outputFile << std::setprecision(12) << piv_data.U[pass][index] << "," << piv_data.V[pass][index] << "\n";
        }
    }
    outputFile << "\n\n\n\n\n";
}
