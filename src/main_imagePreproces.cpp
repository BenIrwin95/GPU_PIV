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

    ImageData im = readTiffToAppropriateIntegerVector("./example_resources/cam1_im_000_A.tiff");


    const std::string inputFile = "./example_resources/IM_FILTER_setup.in";

    // extract non-optional inputs
    int N_filters;
    try{
        N_filters = findIntegerAfterKeyword(inputFile, "N_FILTER");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // create filters
    std::vector<ImFilter> filter_list(N_filters);
    for(int i=0;i<N_filters;i++){
        std::string search_term = fmt::format("FILTER_{}", i);
        try{
            std::string line_string = findRestOfLineAfterKeyword(inputFile, search_term);
            std::vector<std::string> line_words = separate_words(line_string);
            filter_list[i] = createFilter(line_words, env);
        } catch (const std::runtime_error& e) {
            std::cerr << e.what() << std::endl;
            std::cout << "Problem loading " << search_term <<std::endl;
            return 1;
        }
    }

    // load image

    env.im1 = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*im.pixelBytes, NULL, &err); if(err != CL_SUCCESS){return 1;}
    env.im1_complex = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return 1;}
    err = uploadImage_and_convert_to_complex(im, env, env.im1, env.im1_complex); if(err != CL_SUCCESS){return 1;}

    for(int i=0;i<N_filters;i++){
        err = runFilter(env.im1_complex, im, filter_list[i], env); if(err != CL_SUCCESS){return 1;}
    }


    //
    //
    // manual_range_scaling(env.im1_complex, im, 0.0, 0.5, env);

    err =  retrieveImageFromBuffer(env.im1_complex, env.im1, im, env); if(err != CL_SUCCESS){return 1;}



    writeTiffFromAppropriateIntegerVector(im, "./example_resources/test.tiff");

    return 0;
}
