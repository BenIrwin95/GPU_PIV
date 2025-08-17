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




__kernel void mean_filter(__global float2* im, int2 imDim, int radius){
    const int gid = get_global_id(0);
    const int N = imDim.x*imDim.y;
    if(gid < N){
        int i_ref = (int) floor((float) gid/imDim.x);
        int j_ref = gid % imDim.x;
        float val = 0.0f;
        int valCount = 0;
        for(int i=-radius;i<radius;i++){
            for(int j=-radius;j<radius;j++){
                int i_ = i_ref + i;
                int j_ = j_ref + j;
                if(i_ > 0 && i_ < imDim.y && j_ > 0 && j_ < imDim.x){
                    val += im[i_*imDim.x + j_].x;
                    valCount++;
                }
            }
        }
        if(valCount != 0){
            im[i_ref*imDim.x+j_ref].x = val/valCount;
        }
    }
}

__kernel void mean_filter_subtraction(__global float2* im, int2 imDim, int radius){
    const int gid = get_global_id(0);
    const int N = imDim.x*imDim.y;
    if(gid < N){
        int i_ref = (int) floor((float) gid/imDim.x);
        int j_ref = gid % imDim.x;
        float val = 0.0f;
        int valCount = 0;
        for(int i=-radius;i<radius;i++){
            for(int j=-radius;j<radius;j++){
                int i_ = i_ref + i;
                int j_ = j_ref + j;
                if(i_ > 0 && i_ < imDim.y && j_ > 0 && j_ < imDim.x){
                    val += im[i_*imDim.x + j_].x;
                    valCount++;
                }
            }
        }
        if(valCount != 0){
            float temp = im[i_ref*imDim.x+j_ref].x - val/valCount;
            if(temp < 0){temp=0;}
            im[i_ref*imDim.x+j_ref].x = temp;
        }
    }
}






__kernel void gaussian_filter(__global float2* im, int2 imDim, int radius, float stdDev){
    const int gid = get_global_id(0);
    const int N = imDim.x*imDim.y;
    if(gid < N){
        int i_ref = (int) floor((float) gid/imDim.x);
        int j_ref = gid % imDim.x;
        float val = 0.0f;
        float sumVal = 0.0f;
        for(int i=-radius;i<radius;i++){
            for(int j=-radius;j<radius;j++){
                int i_ = i_ref + i;
                int j_ = j_ref + j;
                if(i_ > 0 && i_ < imDim.y && j_ > 0 && j_ < imDim.x){
                    float G = (1.0f/(2.0f * M_PI_F*stdDev*stdDev)) * exp( -(i*i + j*j)/(2.0f*stdDev*stdDev) );
                    val += im[i_*imDim.x + j_].x * G;
                    sumVal += G;
                }
            }
        }
        if(sumVal != 0.0f){
            im[i_ref*imDim.x+j_ref].x = val/sumVal;
        }
    }
}

__kernel void gaussian_filter_subtraction(__global float2* im, int2 imDim, int radius, float stdDev){
    const int gid = get_global_id(0);
    const int N = imDim.x*imDim.y;
    if(gid < N){
        int i_ref = (int) floor((float) gid/imDim.x);
        int j_ref = gid % imDim.x;
        float val = 0.0f;
        float sumVal = 0.0f;
        for(int i=-radius;i<radius;i++){
            for(int j=-radius;j<radius;j++){
                int i_ = i_ref + i;
                int j_ = j_ref + j;
                if(i_ > 0 && i_ < imDim.y && j_ > 0 && j_ < imDim.x){
                    float G = (1.0f/(2.0f * M_PI_F*stdDev*stdDev)) * exp( -(i*i + j*j)/(2.0f*stdDev*stdDev) );
                    val += im[i_*imDim.x + j_].x * G;
                    sumVal += G;
                }
            }
        }
        if(sumVal != 0.0f){
            float temp = im[i_ref*imDim.x+j_ref].x - val/sumVal;
            if(temp<0){temp=0;}
            im[i_ref*imDim.x+j_ref].x = temp;
        }
    }
}


)";





ImFilter createFilter(std::vector<std::string>& words, OpenCL_env& env){
    ImFilter filter;
    filter.name = words[0];
    if(filter.name == "MANUAL_STRETCH"){
        filter.float_args.resize(2);
        filter.float_args[0] = std::stof(words[1]);
        filter.float_args[1] = std::stof(words[2]);
    } else if (filter.name == "MEAN_FILTER") {
        filter.int_args.resize(1);
        filter.int_args[0] = std::stoi(words[1]);
    } else if (filter.name == "MEAN_FILTER_SUBTRACTION") {
        filter.int_args.resize(1);
        filter.int_args[0] = std::stoi(words[1]);
    } else if (filter.name == "GAUSS_FILTER") {
        filter.int_args.resize(1);
        filter.float_args.resize(1);
        filter.int_args[0] = std::stoi(words[1]);
        filter.float_args[0] = std::stoi(words[1]);
    } else if (filter.name == "GAUSS_FILTER_SUBTRACTION") {
        filter.int_args.resize(1);
        filter.float_args.resize(1);
        filter.int_args[0] = std::stoi(words[1]);
        filter.float_args[0] = std::stoi(words[1]);
    } else {
        throw std::invalid_argument("Unknown filter type: " + filter.name);
    }
    return filter;
}

std::vector<ImFilter> create_filter_list(int N_filters, std::string inputFile, OpenCL_env& env){
    std::vector<ImFilter> filter_list(N_filters);
    for(int i=0;i<N_filters;i++){
        std::string search_term = fmt::format("FILTER_{}", i);
        try{
            std::string line_string = findRestOfLineAfterKeyword(inputFile, search_term);
            std::vector<std::string> line_words = separate_words(line_string);
            filter_list[i] = createFilter(line_words, env);
        } catch (const std::runtime_error& e) {
            throw std::invalid_argument("Problem loading " + search_term);
        }
    }
    return filter_list;
}



cl_int runFilter(cl::Buffer& im_buffer_complex, ImageData& im, ImFilter& filter, OpenCL_env& env){
    cl_int err=CL_SUCCESS;
    if(filter.name == "MANUAL_STRETCH"){
        err = manual_range_scaling(im_buffer_complex, im, filter.float_args[0], filter.float_args[1], env);
    } else if(filter.name == "MEAN_FILTER"){
        err = image_mean_filter(im_buffer_complex, im, filter.int_args[0], env);
    } else if(filter.name == "MEAN_FILTER_SUBTRACTION"){
        err = image_mean_filter_subtraction(im_buffer_complex, im, filter.int_args[0], env);
    } else if(filter.name == "GAUSS_FILTER"){
        err = image_gauss_filter(im_buffer_complex, im, filter.int_args[0], filter.float_args[0], env);
    } else if(filter.name == "GAUSS_FILTER_SUBTRACTION"){
        err = image_gauss_filter_subtraction(im_buffer_complex, im, filter.int_args[0], filter.float_args[0], env);
    } else{
        std::cout << "Unknown filter" << std::endl;
        return CL_INVALID_VALUE;
    }


    return err;
}



cl_int process_image_with_filterList(cl::Buffer& im_buffer_complex, ImageData& im, std::vector<ImFilter>& filter_list, OpenCL_env& env){
    cl_int err=CL_SUCCESS;
    for(int i=0;i<filter_list.size();i++){
        err = runFilter(im_buffer_complex, im, filter_list[i], env); if(err != CL_SUCCESS){return err;}
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




cl_int image_mean_filter(cl::Buffer& im_buffer_complex, ImageData& im, int radius, OpenCL_env& env){
    cl_int err=CL_SUCCESS;

    cl_int2 imDim;
    imDim.s[0] = im.width;imDim.s[1] = im.height;
    int N = im.width*im.height;
    try {err = env.kernel_mean_filter.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_mean_filter.setArg(1, sizeof(cl_int2),&imDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_mean_filter.setArg(2, sizeof(int),&radius);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_mean_filter, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_mean_filter" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}





cl_int image_mean_filter_subtraction(cl::Buffer& im_buffer_complex, ImageData& im, int radius, OpenCL_env& env){
    cl_int err=CL_SUCCESS;

    cl_int2 imDim;
    imDim.s[0] = im.width;imDim.s[1] = im.height;
    int N = im.width*im.height;
    try {err = env.kernel_mean_filter_subtraction.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_mean_filter_subtraction.setArg(1, sizeof(cl_int2),&imDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_mean_filter_subtraction.setArg(2, sizeof(int),&radius);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_mean_filter_subtraction, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_mean_filter_subtraction" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}






cl_int image_gauss_filter(cl::Buffer& im_buffer_complex, ImageData& im, int radius, float stdDev, OpenCL_env& env){
    cl_int err=CL_SUCCESS;

    cl_int2 imDim;
    imDim.s[0] = im.width;imDim.s[1] = im.height;
    int N = im.width*im.height;
    try {err = env.kernel_gauss_filter.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter.setArg(1, sizeof(cl_int2),&imDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter.setArg(2, sizeof(int),&radius);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter.setArg(3, sizeof(float),&stdDev);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_gauss_filter, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_gauss_filter" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}




cl_int image_gauss_filter_subtraction(cl::Buffer& im_buffer_complex, ImageData& im, int radius, float stdDev, OpenCL_env& env){
    cl_int err=CL_SUCCESS;

    cl_int2 imDim;
    imDim.s[0] = im.width;imDim.s[1] = im.height;
    int N = im.width*im.height;
    try {err = env.kernel_gauss_filter_subtraction.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter_subtraction.setArg(1, sizeof(cl_int2),&imDim);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter_subtraction.setArg(2, sizeof(int),&radius);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_gauss_filter_subtraction.setArg(3, sizeof(float),&stdDev);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_gauss_filter_subtraction, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_gauss_filter_subtraction" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}
