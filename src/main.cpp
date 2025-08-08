#include "standardHeader.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) { // argc counts all inputs including the .exe name
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    //char *inputFile = argv[1];
    const std::string inputFile = argv[1];
    PIVdata piv_data;


    // extract non-optional inputs
    int DEBUG_LVL;
    int N_frames;
    std::string im1_filepath_template;
    int im1_frame_start;
    int im1_frame_step;
    std::string im2_filepath_template;
    int im2_frame_start;
    int im2_frame_step;
    try{
        DEBUG_LVL = findIntegerAfterKeyword(inputFile, "DEBUG");
        N_frames = findIntegerAfterKeyword(inputFile, "N_FRAMES");
        im1_filepath_template = findRestOfLineAfterKeyword(inputFile,"IMAGEFILE_1");
        im1_frame_start = findIntegerAfterKeyword(inputFile, "IM1_FRAME_START");
        im1_frame_step = findIntegerAfterKeyword(inputFile, "IM1_FRAME_STEP");
        im2_filepath_template = findRestOfLineAfterKeyword(inputFile,"IMAGEFILE_2");
        im2_frame_start = findIntegerAfterKeyword(inputFile, "IM2_FRAME_START");
        im2_frame_step = findIntegerAfterKeyword(inputFile, "IM2_FRAME_STEP");
        piv_data.N_pass = findIntegerAfterKeyword(inputFile, "N_PASS");
        piv_data.window_sizes = findIntegersAfterKeyword(inputFile, "WINDOW_SIZE");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }


    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //---------------------initialise OpenCL-----------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------

    cl_int err;
    OpenCL_env env;
    if(env.status != CL_SUCCESS){
        CHECK_CL_ERROR(env.status);
        return 1;
    }



    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //---------------------Load a single image---------------------------
    //--------------------and initialise memory--------------------------
    //-------------------------------------------------------------------


    // load first image to find its size and data type
    ImageData im_ref = readTiffToAppropriateIntegerVector(fmt::format(fmt::runtime(im1_filepath_template), 1));
    //std::cout << fmt::format(fmt::runtime(im1_filepath_template),1) << " loaded. Size: " << im1.width << "x" << im1.height << std::endl;

    // iterate through the passes to see what the size of our data structures will be
    piv_data.arrSize.resize(piv_data.N_pass);
    piv_data.window_shifts.resize(piv_data.N_pass);
    piv_data.X.resize(piv_data.N_pass);
    piv_data.Y.resize(piv_data.N_pass);
    piv_data.U.resize(piv_data.N_pass);
    piv_data.V.resize(piv_data.N_pass);
    for(int k=0;k<piv_data.N_pass;k++){
        int windowSize = piv_data.window_sizes[k];
        double overlap = 0.5;
        int window_shift = (1.0-overlap)*windowSize;
        piv_data.window_shifts[k] = window_shift;
        piv_data.arrSize[k].s[0] = floor((im_ref.width-windowSize)/window_shift);
        piv_data.arrSize[k].s[1] = floor((im_ref.height-windowSize)/window_shift);
        uint32_t arrLen = piv_data.arrSize[k].s[0] * piv_data.arrSize[k].s[1];
        piv_data.X[k].resize(arrLen);
        piv_data.Y[k].resize(arrLen);
        piv_data.U[k].resize(arrLen);
        piv_data.V[k].resize(arrLen);
        // populate grids
        for(int i=0;i<piv_data.arrSize[k].s[1];i++){
            for(int j=0;j<piv_data.arrSize[k].s[0];j++){
                piv_data.X[k][i*piv_data.arrSize[k].s[0] + j] = (float)windowSize/2 + j*window_shift;
                piv_data.Y[k][i*piv_data.arrSize[k].s[0] + j] = (float)windowSize/2 + i*window_shift;
            }
        }

    }
    err = inititialise_OpenCL_buffers(env, piv_data, im_ref);
    if(err != CL_SUCCESS){
        CHECK_CL_ERROR(err);
        return 1;
    }


    for(int frame=0;frame<N_frames;frame++){
        // load images, upload to GPU and convert to complex format ready for FFT later on
        ImageData im1 = readTiffToAppropriateIntegerVector(fmt::format(fmt::runtime(im1_filepath_template), im1_frame_start + frame*im1_frame_step));
        ImageData im2 = readTiffToAppropriateIntegerVector(fmt::format(fmt::runtime(im2_filepath_template), im2_frame_start + frame*im2_frame_step));

        err = uploadImage_and_convert_to_complex(im1, env, env.im1, env.im1_complex);
        if(err != CL_SUCCESS){CHECK_CL_ERROR(err);std::cout << fmt::format(fmt::runtime(im1_filepath_template), im1_frame_start + frame*im1_frame_step) <<" could not be uploaded" << std::endl;break;}
        err = uploadImage_and_convert_to_complex(im2, env, env.im2, env.im2_complex);
        if(err != CL_SUCCESS){CHECK_CL_ERROR(err);std::cout << fmt::format(fmt::runtime(im2_filepath_template), im2_frame_start + frame*im2_frame_step) <<" could not be uploaded" << std::endl;break;}


        // iterate through passes with different window sizes
        for(int pass=0;pass<piv_data.N_pass;pass++){
            // upload correct X and Y grid for this pass
            const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
            try{
                env.queue.enqueueWriteBuffer( env.X, CL_TRUE, 0, gridSizeBytes, piv_data.X[pass].data());
                env.queue.enqueueWriteBuffer( env.Y, CL_TRUE, 0, gridSizeBytes, piv_data.Y[pass].data());
            } catch (cl::Error& e) {
                std::cerr << "Error uploading to GPU" << std::endl;
                CHECK_CL_ERROR(e.err());
                return 1;
            }


            // divide image into tiles
            err = uniformly_tile_data(env.im1_complex, im1.dims, env.im1_windows, piv_data.window_sizes[pass], piv_data.window_shifts[pass], piv_data.arrSize[pass], env);
            if(err != CL_SUCCESS){CHECK_CL_ERROR(err);break;}
            err = uniformly_tile_data(env.im2_complex, im2.dims, env.im2_windows, piv_data.window_sizes[pass], piv_data.window_shifts[pass], piv_data.arrSize[pass], env);
            if(err != CL_SUCCESS){CHECK_CL_ERROR(err);break;}


        }





    }

    return 0;
}

// // 1. Get a platform and device
// std::vector<cl::Platform> platforms;
// cl::Platform::get(&platforms);
// cl::Platform platform = platforms[0];
//
// std::vector<cl::Device> devices;
// platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
// cl::Device device = devices[0];
//
// // 2. Create context and command queue
// cl::Context context(device);
// cl::CommandQueue queue(context, device);
//
// // 3. Create host data
// int n = 1024;
// std::vector<float> a(n, 1.0f);
// std::vector<float> b(n, 2.0f);
// std::vector<float> c(n);
//
// // 4. Create buffers on the device
// cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
// cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
// cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);
//
// // 5. Build the program and kernel
// std::string kernelCode =
// "__kernel void add_vectors(__global const float* a, __global const float* b, __global float* c) {"
// "    int gid = get_global_id(0);"
// "    c[gid] = a[gid] + b[gid];"
// "}";
// cl::Program program(context, kernelCode);
// program.build(device);
// cl::Kernel kernel(program, "add_vectors");
//
// // 6. Set kernel arguments
// kernel.setArg(0, bufferA);
// kernel.setArg(1, bufferB);
// kernel.setArg(2, bufferC);
//
// // 7. Execute the kernel
// cl::NDRange global(n);
// cl::NDRange local(256);
// queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
//
// // 8. Read the results back to the host
// queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * n, c.data());
//
// // 9. Verify the results
// bool correct = true;
// for (int i = 0; i < n; ++i) {
//     if (c[i] != 3.0f) {
//         correct = false;
//         break;
//     }
// }
// std::cout << "Result is " << (correct ? "correct" : "incorrect") << std::endl;
