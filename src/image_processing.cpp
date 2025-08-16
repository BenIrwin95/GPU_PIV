#include "standardHeader.hpp"


const std::string kernelSource_image_processing=R"(

__kernel void manual_range_scaling(__global float2* im, int N, float minVal, float maxVal, float absolute_max){
    // this should set everything below minVal to zero and everything above maxVal to absolute_max
    const int gid = get_global_id(0);
    if(gid < N){
        float pixelVal = im[gid].x;
        pixelVal -= minVal;
        pixelVal = absolute_max * pixelVal/(maxVal - minVal);
        if(pixelVal > absolute_max){
            pixelVal = absolute_max;
        }
        im[gid].x = pixelVal;
    }
}


)";





ImFilter createFilter(std::vector<std::string>& words, OpenCL_env& env){
    ImFilter filter;
    filter.name = words[0];
    if(filter.name == "MANUAL_STRETCH"){
        filter.kernel = env.kernel_manual_range_scaling;
        filter.float_args.resize(2);
        filter.float_args[0] = std::stof(words[1]);
        filter.float_args[1] = std::stof(words[2]);
    } else {
        throw std::invalid_argument("Unknown filter type: " + filter.name);
    }
    return filter;
}


cl_int runFilter(cl::Buffer& im_buffer_complex, ImageData& im, ImFilter& filter, OpenCL_env& env){
    cl_int err=CL_SUCCESS;
    if(filter.name == "MANUAL_STRETCH"){
        err = manual_range_scaling(im_buffer_complex, im, filter.float_args[0], filter.float_args[1], env);
    } else{
        std::cout << "Unknown filter" << std::endl;
        return CL_INVALID_VALUE;
    }


    return err;
}





cl_int manual_range_scaling(cl::Buffer& im_buffer_complex, ImageData& im, float minVal, float maxVal, OpenCL_env& env){
    // min and max val will be on scale of 0 to 1 to represent whatever the datatypes min and max vals are
    cl_int err =CL_SUCCESS;

    float absolute_max;
    switch (im.type) {
        case ImageData::DataType::UINT8:
            absolute_max = (float) std::numeric_limits<uint8_t>::max();
            break;
        case ImageData::DataType::UINT16:
            absolute_max = (float) std::numeric_limits<uint16_t>::max();
            break;
        case ImageData::DataType::UINT32:
            absolute_max = (float) std::numeric_limits<uint32_t>::max();
            break;
        case ImageData::DataType::UNKNOWN:
        default:
            std::cout << "Data type is unknown or unsupported." << std::endl;
            return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }

    minVal = minVal * absolute_max;
    maxVal = maxVal * absolute_max;

    int N = im.width*im.height;
    try {err = env.kernel_manual_range_scaling.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_manual_range_scaling.setArg(1, sizeof(int),&N);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_manual_range_scaling.setArg(2, sizeof(int),&minVal);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_manual_range_scaling.setArg(3, sizeof(int),&maxVal);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_manual_range_scaling.setArg(4, sizeof(int),&absolute_max);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_manual_range_scaling, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_manual_range_scaling" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();



    return err;

}
