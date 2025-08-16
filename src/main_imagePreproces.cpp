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

    env.im1 = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*im.pixelBytes, NULL, &err); if(err != CL_SUCCESS){return 1;}
    env.im1_complex = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return 1;}

    err = uploadImage_and_convert_to_complex(im, env, env.im1, env.im1_complex);


    manual_range_scaling(env.im1_complex, im, 0.0, 0.5, env);

    //std::vector<cl_float2> im_complex_host(im.width*im.height);
    // env.queue.enqueueReadBuffer(env.im1_complex, CL_TRUE, 0, im.width*im.height*sizeof(cl_float2), im_complex_host.data());
    err =  retrieveImageFromBuffer(env.im1_complex, env.im1, im, env); if(err != CL_SUCCESS){return 1;}



    writeTiffFromAppropriateIntegerVector(im, "./example_resources/test.tiff");

    return 0;
}
