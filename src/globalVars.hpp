#ifndef GLOBAL_VARS_HPP
#define GLOBAL_VARS_HPP

#include "standardLibraries.hpp"


struct PIVdata {
    int N_pass;
    std::vector<int> window_sizes;
};



struct OpenCL_env {
    cl::Platform platform;            // OpenCL platform
    cl::Device device_id;             // device ID
    cl::Context context;                 // context
    cl::CommandQueue queue;             // command queue
    cl::Program program;                 // program
    cl_int status;


    // constructor (this will automatically setup the OpenCL environment for this project)
    OpenCL_env() {
        cl_int err; // Internal variable for error codes

        // Get platform
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        if (err != CL_SUCCESS) {
            std::cerr << "Error getting platform: " << err << std::endl;
            status=err;
            return; //throw exception
        }
        platform = platforms[0];

        // Get device
        std::vector<cl::Device> devices;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (err != CL_SUCCESS) {
            std::cerr << "Error getting device: " << err << std::endl;
            status=err;
            return;
        }
        device_id = devices[0];

        // Create context
        context = cl::Context(device_id, nullptr, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating context: " << err << std::endl;
            status=err;
            return;
        }

        // Create command queue
        queue = cl::CommandQueue(context, device_id, 0, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating command queue: " << err << std::endl;
            status=err;
            return;
        }

        // will need to later something for building program
        status=CL_SUCCESS;
        std::cout << "OpenCL environment successfully initialized." << std::endl;
    }

};


#endif
