#include "standardHeader.hpp"
#include <algorithm>
#include <numeric>

/*
 * Page 159 of Particle Image Velocimetry textbook
 * Filters to create
 * - manual range clipping
 * - median range clipping
 * - dynamic histogram stretching
 * - localised background subtraction
*/


// Find the mean of a vector of any numeric type
template <typename T>
double findMean(const std::vector<T>& data) {
    if (data.empty()) {
        return 0.0;
    }
    // Use double for the accumulator to prevent overflow and ensure a floating-point result
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Find the median of a vector of any numeric type
template <typename T>
double findMedian(std::vector<T> data) {
    if (data.empty()) {
        return 0.0;
    }
    std::sort(data.begin(), data.end());
    size_t size = data.size();
    if (size % 2 == 1) {
        return static_cast<double>(data[size / 2]);
    } else {
        return (static_cast<double>(data[size / 2 - 1]) + static_cast<double>(data[size / 2])) / 2.0;
    }
}

// Find the standard deviation of a vector of any numeric type
template <typename T>
double findStdDev(const std::vector<T>& data) {
    if (data.size() <= 1) {
        return 0.0;
    }
    double mean = findMean(data); // The templated findMean is used here
    double sumOfSquaredDiff = 0.0;
    for (const T& val : data) {
        sumOfSquaredDiff += std::pow(static_cast<double>(val) - mean, 2);
    }
    return std::sqrt(sumOfSquaredDiff / (data.size() - 1));
}


template <typename T>
int findPercentileValue(std::vector<T> data, double percentile) {
    if (data.empty()) {
        return 0; // Or handle error appropriately
    }

    // Sort the vector in ascending order
    std::sort(data.begin(), data.end());



    // Calculate the index for the given percentile
    double index_float = percentile * (data.size() - 1);

    // Linearly interpolate between the two nearest integer indices
    int index_low = static_cast<int>(floor(index_float));
    int index_high = static_cast<int>(ceil(index_float));

    if (index_low == index_high) {
        return data[index_low];
    }

    double value_low = static_cast<double>(data[index_low]);
    double value_high = static_cast<double>(data[index_high]);

    return static_cast<int>(value_low + (value_high - value_low) * (index_float - index_low));
}




int main(){

    cl_int err;
    OpenCL_env env;
    if(env.status != CL_SUCCESS){CHECK_CL_ERROR(env.status);return 1;}




    const std::string inputFile = "./example_resources/IM_FILTER_setup.in";

    // extract non-optional inputs
    int N_filters;
    std::string input_im_file;
    std::string output_im_file;
    try{
        N_filters = findIntegerAfterKeyword(inputFile, "N_FILTER");
        input_im_file = findRestOfLineAfterKeyword(inputFile,"IMAGE_SRC");
        output_im_file = findRestOfLineAfterKeyword(inputFile,"IMAGE_DST");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }


    ImageData im = readTiffToAppropriateIntegerVector(input_im_file);


    // calculate and display some statistics about the image
    // Use std::visit to call findMean on the active type in the variant
    double mean = std::visit([](const auto& data) {return findMean(data);}, im.pixelData);
    double median = std::visit([](const auto& data) {return findMedian(data);}, im.pixelData);
    double std_dev = std::visit([](const auto& data) {return findStdDev(data);}, im.pixelData);

    int percentile_1 = std::visit([](auto& data) {return findPercentileValue(data, 0.01);}, im.pixelData);
    int percentile_99 = std::visit([](auto& data) {return findPercentileValue(data, 0.99);}, im.pixelData);



    // find the max value for this datatype
    double absolute_max;
    switch (im.type) {
        case ImageData::DataType::UINT8:
            absolute_max = (double) std::numeric_limits<uint8_t>::max();
            break;
        case ImageData::DataType::UINT16:
            absolute_max = (double) std::numeric_limits<uint16_t>::max();
            break;
        case ImageData::DataType::UINT32:
            absolute_max = (double) std::numeric_limits<uint32_t>::max();
            break;
        case ImageData::DataType::UNKNOWN:
        default:
            std::cout << "Image data type is unknown or unsupported." << std::endl;
            return 1;
    }
    mean = 100 * mean/absolute_max;
    median = 100 * median/absolute_max;
    std_dev = 100 * std_dev/absolute_max;
    double percentile_1_double = 100 * (double)percentile_1/absolute_max;
    double percentile_99_double = 100 * (double)percentile_99/absolute_max;


    std::cout << "Mean pixel intensity: " << mean << "%" << std::endl;
    std::cout << "Median pixel intensity: " << median << "%" << std::endl;
    std::cout << "StdDev pixel intensity: " << std_dev << "%" << std::endl;
    std::cout << "Threshold of 1\% of all pixel intensities: " << percentile_1_double << "%" << std::endl;
    std::cout << "Threshold of 99\% of all pixel intensities: " << percentile_99_double << "%" << std::endl;


    // create filters
    std::vector<ImFilter> filter_list = create_filter_list(N_filters, inputFile, env);


    // load image

    env.im1 = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*im.pixelBytes, NULL, &err); if(err != CL_SUCCESS){return 1;}
    env.im1_complex = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return 1;}
    err = uploadImage_and_convert_to_complex(im, env, env.im1, env.im1_complex); if(err != CL_SUCCESS){return 1;}


    err = process_image_with_filterList(env.im1_complex, im, filter_list, env); if(err != CL_SUCCESS){return 1;}


    err =  retrieveImageFromBuffer(env.im1_complex, env.im1, im, env); if(err != CL_SUCCESS){return 1;}



    writeTiffFromAppropriateIntegerVector(im, output_im_file);

    return 0;
}
