#ifndef GLOBAL_VARS_HPP
#define GLOBAL_VARS_HPP

#include "standardLibraries.hpp"


struct PIVdata {
    int N_pass;
    std::vector<int> window_sizes;
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> Y;
    std::vector<std::vector<float>> U;
    std::vector<std::vector<float>> V;
    std::vector<cl_int2> arrSize;
};




struct ImageData {
    uint32_t width;
    uint32_t height;

    // std::variant can hold one of these vector types at any given time.
    // This function specifically handles unsigned integer types.
    std::variant<std::vector<uint8_t>, std::vector<uint16_t>, std::vector<uint32_t>> pixelData;


    // Enum to indicate which type is currently stored in pixelData
    enum class DataType {
        UINT8,
        UINT16,
        UINT32,
        UNKNOWN // Should not be reached if checks are robust
    } type;
    size_t pixelBytes; // bytes in a single pixel
    size_t sizeBytes; //total size of image in bytes
};



struct OpenCL_env {
    cl::Platform platform;            // OpenCL platform
    cl::Device device_id;             // device ID
    cl::Context context;                 // context
    cl::CommandQueue queue;             // command queue
    cl::Program program;                 // program
    cl_int status;
    // memory structures for working on GPU
    cl::Buffer im1;
    cl::Buffer im2;
    cl::Buffer im1_windows;
    cl::Buffer im2_windows;
    cl::Buffer X;
    cl::Buffer Y;
    cl::Buffer U;
    cl::Buffer V;


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
