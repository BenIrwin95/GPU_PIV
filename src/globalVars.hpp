#ifndef GLOBAL_VARS_HPP
#define GLOBAL_VARS_HPP

#include "standardLibraries.hpp"


extern const std::string kernelSource_tiffFunctions;
extern const std::string kernelSource_dataArrangement;
extern const std::string kernelSource_FFT;
extern const std::string kernelSource_complexMaths;
extern const std::string kernelSource_determineCorrelation;
extern const std::string kernelSource_vectorValidation;
extern const std::string kernelSource_bicubic_interpolation;
extern const std::string kernelSource_image_processing;






struct PIVdata {
    int N_pass;
    int N_frames;
    std::vector<int> window_sizes;
    std::vector<int> window_shifts;
    std::vector<float> window_overlaps;
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> Y;
    std::vector<std::vector<float>> U;
    std::vector<std::vector<float>> V;
    // also need data in double form for spline interpolation
    std::vector<std::vector<float>> x; // x and y are just a 1d array of the x and y coordinates along each side. They are NOT a flattened 2D vector
    std::vector<std::vector<float>> y;
    std::vector<cl_int2> arrSize;
};



struct ImageData {
    uint32_t width;
    uint32_t height;
    cl_int2 dims;

    // std::variant can hold one of these vector types at any given time.
    // This function specifically handles unsigned integer types.
    std::variant<std::vector<uint8_t>, std::vector<uint16_t>, std::vector<uint32_t>> pixelData;
    std::vector<cl_float2> complexPixelData;


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

struct ImFilter {
    std::string name;
    cl::Kernel kernel;
    std::vector<float> float_args;
    std::vector<int> int_args;
};





struct OpenCL_env {
    cl::Platform platform;            // OpenCL platform
    cl::Device device_id;             // device ID
    cl::Context context;                 // context
    cl::CommandQueue queue;             // command queue
    cl::Program program;                 // program
    cl_int status;
    // kernels
    cl::Kernel kernel_convert_im_to_complex;
    cl::Kernel kernel_convert_im_to_complex_uint16;
    cl::Kernel kernel_convert_im_to_complex_uint32;
    cl::Kernel kernel_bicubic_interpolation;
    cl::Kernel kernel_uniform_tiling;
    cl::Kernel kernel_warped_tiling;
    cl::Kernel kernel_detrend_window;
    cl::Kernel kernel_FFT_1D;
    cl::Kernel kernel_complex_multiply_conjugate_norm;
    cl::Kernel kernel_findMaxCorr;
    cl::Kernel kernel_identifyInvalidVectors;
    cl::Kernel kernel_correctInvalidVectors;
    // image processing kernels
    cl::Kernel kernel_convert_float2_to_uint8;
    cl::Kernel kernel_convert_float2_to_uint16;
    cl::Kernel kernel_convert_float2_to_uint32;
    cl::Kernel kernel_manual_range_scaling;
    cl::Kernel kernel_mean_filter;

    // memory structures for working on GPU
    cl::Buffer im1;
    cl::Buffer im2;
    cl::Buffer im1_complex;
    cl::Buffer im2_complex;
    cl::Buffer im1_windows;
    cl::Buffer im2_windows;
    cl::Buffer X;
    cl::Buffer Y;
    cl::Buffer U;
    cl::Buffer V;
    cl::Buffer flags;
    // bicubic interpolation
    cl::Buffer x_vals;
    cl::Buffer y_vals;
    cl::Buffer x_vals_im;
    cl::Buffer y_vals_im;
    cl::Buffer x_ref;
    cl::Buffer y_ref;
    cl::Buffer U_ref;
    cl::Buffer V_ref;
    cl::Buffer imageShifts_x;
    cl::Buffer imageShifts_y;


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

        // loading kernels sources into program
        try {
            cl::Program::Sources sources;
            sources.push_back({kernelSource_complexMaths.c_str(), kernelSource_complexMaths.length()});
            sources.push_back({kernelSource_tiffFunctions.c_str(), kernelSource_tiffFunctions.length()});
            sources.push_back({kernelSource_dataArrangement.c_str(), kernelSource_dataArrangement.length()});
            sources.push_back({kernelSource_FFT.c_str(), kernelSource_FFT.length()});
            sources.push_back({kernelSource_determineCorrelation.c_str(), kernelSource_determineCorrelation.length()});
            sources.push_back({kernelSource_vectorValidation.c_str(), kernelSource_vectorValidation.length()});
            sources.push_back({kernelSource_bicubic_interpolation.c_str(), kernelSource_bicubic_interpolation.length()});
            sources.push_back({kernelSource_image_processing.c_str(), kernelSource_image_processing.length()});
            program = cl::Program(context, sources);

        } catch (cl::Error& e) {
            status=e.err();
            std::cerr << "Error creating program: " << e.what() << " (" << e.err() << ")" << std::endl;
            return;
        }


        // build program
        try {
            program.build(devices);
        } catch (cl::Error& e) {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                // Find the device that failed to build
                for (const auto& device : devices) {
                    // Get the build log for the device
                    std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
                    std::cerr << log << std::endl;
                }
            }
            status = e.err();
            return;
        }

        // kernel creation
        try {
            kernel_convert_im_to_complex = cl::Kernel(program, "convert_to_float2");
            kernel_convert_im_to_complex_uint16 = cl::Kernel(program, "convert_uint16_to_float2");
            kernel_convert_im_to_complex_uint32 = cl::Kernel(program, "convert_uint32_to_float2");
            kernel_bicubic_interpolation = cl::Kernel(program, "bicubic_interpolation");
            kernel_uniform_tiling = cl::Kernel(program, "uniform_tiling");
            kernel_warped_tiling = cl::Kernel(program, "warped_tiling");
            kernel_detrend_window = cl::Kernel(program, "detrend_window");
            kernel_FFT_1D = cl::Kernel(program, "FFT_1D");
            kernel_complex_multiply_conjugate_norm = cl::Kernel(program, "complex_multiply_conjugate_norm");
            kernel_findMaxCorr = cl::Kernel(program, "findMaxCorr");
            kernel_identifyInvalidVectors = cl::Kernel(program, "identifyInvalidVectors");
            kernel_correctInvalidVectors = cl::Kernel(program, "correctInvalidVectors");
            kernel_convert_float2_to_uint8 = cl::Kernel(program, "convert_float2_to_uint8");
            kernel_convert_float2_to_uint16 = cl::Kernel(program, "convert_float2_to_uint16");
            kernel_convert_float2_to_uint32 = cl::Kernel(program, "convert_float2_to_uint32");
            kernel_manual_range_scaling = cl::Kernel(program, "manual_range_scaling");
            kernel_mean_filter = cl::Kernel(program, "mean_filter");
            // Kernel creation was successful
        } catch (cl::Error& e) {
            std::cerr << "Error creating kernels: " << e.what() << " (" << e.err() << ")" << std::endl;
            status=e.err();
            return;
        }



        //kernel_convert_im_to_complex

        status=CL_SUCCESS;
        std::cout << "OpenCL environment successfully initialized." << std::endl;
    }

};








#endif
