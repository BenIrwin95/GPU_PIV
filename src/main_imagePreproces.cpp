#include "standardHeader.hpp"


/*
 * Page 159 of Particle Image Velocimetry textbook
 * Filters to create
 * - manual range clipping
 * - median range clipping
 * - dynamic histogram stretching
 * - localised background subtraction
*/
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
