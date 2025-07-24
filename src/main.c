#include "err_codes.h"
// standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
// custom headers
#include "tiffFunctions.h" // imports functions for loading and saving tiff images
#include "utilities.h" // imports miscellaneous functions
#include "inputFunctions.h" // imports functions for taking inputs from a text file
#include "determineCorrelation.h"
#include "libGPU_FFT.h"
#include "mainKernel.h"


#define MAX_FILEPATH_LENGTH 200


void printComplexArray(cl_float2* input, size_t width, size_t height){
    printf("\n");
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            int idx = i*width+j;
            printf(" (%.2f, %.2f)",input[idx].x, input[idx].y);
        }
        printf("\n");
    }
}


// main program
int main(int argc, char* argv[]) {
    if (argc != 2) { // argc counts all inputs including the .exe name
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1; // Indicate an error
    }




    clock_t currentTime = clock();
    
    char *inputFile = argv[1];
    int status = INPUTFUNCTIONS_SUCCESS;
    int DEBUG_LVL = extract_int_by_keyword(inputFile, "DEBUG", &status);
    debug_message("Loading input file variables", DEBUG_LVL, 0, &currentTime);

    int N_frames = extract_int_by_keyword(inputFile, "N_FRAMES", &status);

    char *im1_filepath_template = find_line_after_keyword(inputFile, "IMAGEFILE_1", &status);
    int im1_frame_start = extract_int_by_keyword(inputFile, "IM1_FRAME_START", &status);
    int im1_frame_step = extract_int_by_keyword(inputFile, "IM1_FRAME_STEP", &status);
    char *im2_filepath_template = find_line_after_keyword(inputFile, "IMAGEFILE_2", &status);
    int im2_frame_start = extract_int_by_keyword(inputFile, "IM2_FRAME_START", &status);
    int im2_frame_step = extract_int_by_keyword(inputFile, "IM2_FRAME_STEP", &status);

    int N_pass = extract_int_by_keyword(inputFile, "N_PASS", &status);

    int* windowSizes = find_int_list_after_keyword(inputFile, "WINDOW_SIZE", N_pass, &status);


    char *outputFile_template = find_line_after_keyword(inputFile, "OUTPUT_TEMPLATE", &status);

    if(status == INPUTFUNCTIONS_ERR){
        printf("ERROR: Something went wrong loading the input file\n");
        free(im1_filepath_template);free(im2_filepath_template);free(windowSizes);free(outputFile_template);
        return 1;
    }



    //----------------------------------------------------------------------------------------------------
    //-----------------------------------OpenCL initialisation--------------------------------------------
    //----------------------------------------------------------------------------------------------------
    debug_message("Initial OpenCL setup", DEBUG_LVL, 0, &currentTime);

    // General OpenCL setup
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_int err;
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not bind to platform\n");return 1;}
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not get device id\n");return 1;}
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not create context\n");return 1;}
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not create command queue\n");return 1;}


    cl_queue_properties non_blocking_properties[] = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        0 // Must terminate with 0
    };
    cl_command_queue queueNonBlocking = clCreateCommandQueueWithProperties(context, device_id, non_blocking_properties, &err);

    // Create the compute program from the source buffer
    const char* kernel_sources[] = { kernelSource_MaxCorr, kernelSource_complexMaths, kernelSource_FFT_1D };
    program = clCreateProgramWithSource(context, 3, kernel_sources, NULL, &err);
    //free(kernel_sources);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could retrieve source code for kernel\n");return 1;}
    // Build the program executable
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(err != CL_SUCCESS){ // handling errors when compiling the kernel
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *) malloc(log_size + 1); // +1 for null terminator
        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0'; // Null-terminate
        fprintf(stderr, "Kernel Build Error:\n%s\n", build_log);
        free(build_log);
    }
    // Create the compute kernel in the program
    cl_kernel kernelMultConj;
    kernelMultConj = clCreateKernel(program, "complex_multiply_conjugate_norm", &err);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not add kernel function to program\n");return 1;}
    cl_kernel kernelMaxCorr;
    kernelMaxCorr = clCreateKernel(program, "findMaxCorr", &err);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not add kernel function to program\n");return 1;}
    cl_kernel kernelFFT_1D;
    kernelFFT_1D = clCreateKernel(program, "FFT_1D", &err);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not add kernel function to program\n");return 1;}


    cl_ulong local_mem_size;
    err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    if(err!=CL_SUCCESS){printf("ERROR: OpenCL could not determine max local mem size\n");return 1;}



    // initialise buffers
    debug_message("Initialise buffers", DEBUG_LVL, 0, &currentTime);
    cl_mem im1_GPU;
    cl_mem im2_GPU;


    size_t maxTiledInputSize = 0;
    char temp_file[MAX_FILEPATH_LENGTH];
    snprintf(temp_file, sizeof(temp_file),im1_filepath_template, im1_frame_start);
    uint32_t temp_width, temp_height;
    get_tiff_dimensions_single_channel(temp_file, &temp_width, &temp_height);
    for(int i=0;i<N_pass;i++){
        int windowSize = windowSizes[i];
        double overlap = 0.5;
        int window_shift = (1.0-overlap)*windowSize;
        int N_vec_cols = floor(temp_width/window_shift);
        int N_vec_rows = floor(temp_height/window_shift);
        size_t bytesNeeded = (N_vec_cols*windowSize)*(N_vec_rows*windowSize) * sizeof(cl_float2);
        if(bytesNeeded > maxTiledInputSize){
            maxTiledInputSize = bytesNeeded;
        }
    }
    cl_mem im1_windows_max = clCreateBuffer(context, CL_MEM_READ_WRITE, maxTiledInputSize, NULL, &err);
    cl_mem im2_windows_max = clCreateBuffer(context, CL_MEM_READ_WRITE, maxTiledInputSize, NULL, &err);


    debug_message("Iterating through frame-pairs", DEBUG_LVL, 0, &currentTime);
    for(int frame=0;frame<N_frames;frame++){
        char debugMessage[100];
        snprintf(debugMessage, sizeof(debugMessage), "----------------------------------------------\nframe-pair: %d of %d", frame+1, N_frames);
        debug_message(debugMessage, DEBUG_LVL, 1, &currentTime);
        char im1_file[MAX_FILEPATH_LENGTH];
        char im2_file[MAX_FILEPATH_LENGTH];

        snprintf(im1_file, sizeof(im1_file),im1_filepath_template, im1_frame_start + frame*im1_frame_step);
        snprintf(im2_file, sizeof(im2_file),im2_filepath_template, im2_frame_start + frame*im2_frame_step);

        //----------------------------------------------------------------------------------------------------
        //---------------------------------------Loading Images-----------------------------------------------
        //----------------------------------------------------------------------------------------------------
        debug_message("Loading images", DEBUG_LVL, 2, &currentTime);
        // load images
        uint32_t IMWIDTH, IMHEIGHT;
        uint8_t* im1_raw = readSingleChannelTiff(im1_file, &IMWIDTH, &IMHEIGHT);
        uint8_t* im2_raw = readSingleChannelTiff(im2_file, &IMWIDTH, &IMHEIGHT);

        // convert them to complex form
        cl_float2* im1 = tiff2complex(im1_raw ,IMWIDTH, IMHEIGHT);
        cl_float2* im2 = tiff2complex(im2_raw ,IMWIDTH, IMHEIGHT);
        free(im1_raw);free(im2_raw);

        // load the images to GPU
        size_t imageBytes = IMWIDTH*IMHEIGHT*sizeof(cl_float2);
        if(frame==0){
            // no need to repeat this
            im1_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, imageBytes, NULL, &err);
            im2_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, imageBytes, NULL, &err);
        }
        err = clEnqueueWriteBuffer( queue, im1_GPU, CL_TRUE, 0, imageBytes, im1, 0, NULL, NULL );
        err = clEnqueueWriteBuffer( queue, im2_GPU, CL_TRUE, 0, imageBytes, im2, 0, NULL, NULL );



        snprintf(debugMessage, sizeof(debugMessage), "Image loaded: %s \t (%d x %d) pixels",im1_file, IMWIDTH, IMHEIGHT);
        debug_message(debugMessage, DEBUG_LVL, 2, &currentTime);
        snprintf(debugMessage, sizeof(debugMessage), "Image loaded: %s \t (%d x %d) pixels",im2_file, IMWIDTH, IMHEIGHT);
        debug_message(debugMessage, DEBUG_LVL, 2, &currentTime);





        
        debug_message("Preparing savefile", DEBUG_LVL, 2, &currentTime);
        // prepare file for saving
        char output_filename[MAX_FILEPATH_LENGTH];
        snprintf(output_filename, sizeof(output_filename),outputFile_template, frame);
        FILE* fp = fopen(output_filename, "w");
        if (fp == NULL) {
            fprintf(stderr, "Error: Could not open file '%s' for writing.\n", output_filename);
            return 1; // Indicate failure
        }

        for(int pass=0;pass<N_pass;pass++){
            snprintf(debugMessage, sizeof(debugMessage), "Pass %d of %d (window size: %d)",pass+1, N_pass, windowSizes[pass]);
            debug_message(debugMessage, DEBUG_LVL, 2, &currentTime);
            //----------------------------------------------------------------------------------------------------
            //-------------------------------Initialising PIV variables-------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Initialising PIV variables", DEBUG_LVL, 3, &currentTime);
            // determine how many windows will fit across the images
            int windowSize = windowSizes[pass];
            //const int windowSize2 = pow((float)windowSize, 2.0);
            double overlap = 0.5;
            int window_shift = (1.0-overlap)*windowSize;
            int N_vec_cols = floor(IMWIDTH/window_shift);
            int N_vec_rows = floor(IMHEIGHT/window_shift);
            // allocate memory for velocity data
            float* U = (float*)malloc(sizeof(float)*N_vec_rows*N_vec_cols);
            float* V = (float*)malloc(sizeof(float)*N_vec_rows*N_vec_cols);
            float dt = 1.0;
            // allocate memory on GPU for U and V
            size_t vec_bytes = N_vec_cols*N_vec_rows*sizeof(float);
            cl_mem U_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vec_bytes, NULL, &err);
            cl_mem V_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vec_bytes, NULL, &err);
            //----------------------------------------------------------------------------------------------------
            //-------------------------------Initialising windows-------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Initialising windows", DEBUG_LVL, 3, &currentTime);
            cl_mem im1_windows;
            cl_mem im2_windows;
            size_t windowedImageBytes = (N_vec_cols*windowSize)*(N_vec_rows*windowSize) * sizeof(cl_float2);
            //im1_windows = clCreateBuffer(context, CL_MEM_READ_WRITE, windowedImageBytes, NULL, &err);
            //im2_windows = clCreateBuffer(context, CL_MEM_READ_WRITE, windowedImageBytes, NULL, &err);
            cl_buffer_region subRegion;
            subRegion.origin = 0 * sizeof(float); // where we start
            subRegion.size = windowedImageBytes;   // Use a region of 128 floats
            im1_windows = clCreateSubBuffer(im1_windows_max, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &subRegion, NULL);
            im2_windows = clCreateSubBuffer(im2_windows_max, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &subRegion, NULL);

            // The row pitches in bytes
            const size_t src_row_pitch = IMWIDTH * sizeof(cl_float2);
            const size_t src_slice_pitch = imageBytes; // byte size of 2D slice (clEnqueueCopyBufferRect is also meant for 3D)
            const size_t dst_row_pitch = (N_vec_cols*windowSize) * sizeof(cl_float2);
            const size_t dst_slice_pitch = windowedImageBytes;
            // Destination origin (where to start in the destination buffer)
            // The offset in bytes is computed as dst_origin[2] × dst_slice_pitch + dst_origin[1] × dst_row_pitch + dst_origin[0].
            debug_message("Copying windows to tiles", DEBUG_LVL, 4, &currentTime);
            // The dimensions of the region to copy (width, height, depth) --> (width in bytes, height in rows, depth in slices)
            const size_t region[3] = {windowSize*sizeof(cl_float2), windowSize, 1};
            for(int i=0;i<N_vec_rows;i++){
                for(int j=0;j<N_vec_cols;j++){
                    size_t src_origin[3];
                    size_t dst_origin[3];
                    // im1

                    // Source origin (where to start in the source buffer)
                    // The offset in bytes is computed as src_origin[2] × src_slice_pitch + src_origin[1] × src_row_pitch + src_origin[0].
                    //const size_t src_origin[3] = { window_shift*j* sizeof(cl_float2), window_shift * i, 0};
                    src_origin[0] = window_shift*j* sizeof(cl_float2);
                    src_origin[1] = window_shift * i;
                    src_origin[2] = 0.0;
                    dst_origin[0] = windowSize*j* sizeof(cl_float2);
                    dst_origin[1] = windowSize * i;
                    dst_origin[2] = 0.0;
                    // --- Enqueue the GPU-to-GPU Copy ---
                    err = clEnqueueCopyBufferRect(queueNonBlocking,im1_GPU,im1_windows,
                                                  src_origin,dst_origin,
                                                  region,
                                                  src_row_pitch,src_slice_pitch,
                                                  dst_row_pitch,dst_slice_pitch,
                                                  0, NULL, NULL
                    );
                    // im2
                    // Source origin (where to start in the source buffer)
                    // The offset in bytes is computed as src_origin[2] × src_slice_pitch + src_origin[1] × src_row_pitch + src_origin[0].
                    src_origin[0] = window_shift*j* sizeof(cl_float2);
                    src_origin[1] = window_shift * i;
                    src_origin[2] = 0.0;
                    dst_origin[0] = windowSize*j* sizeof(cl_float2);
                    dst_origin[1] = windowSize * i;
                    dst_origin[2] = 0.0;
                    // --- Enqueue the GPU-to-GPU Copy ---
                    err = clEnqueueCopyBufferRect(queueNonBlocking,im2_GPU,im2_windows,
                                                  src_origin,dst_origin,
                                                  region,
                                                  src_row_pitch,src_slice_pitch,
                                                  dst_row_pitch,dst_slice_pitch,
                                                  0, NULL, NULL
                    );
                }
            }
            /* Wait for calculations to be finished. */
            err = clFinish(queueNonBlocking);
            
            
            // find correlations
            debug_message("Calculating correlation", DEBUG_LVL, 3, &currentTime);
            cl_int2 inputDim;
            inputDim.x = N_vec_cols*windowSize;
            inputDim.y = N_vec_rows*windowSize;
            FFT_corr_tiled (im1_windows,im2_windows, inputDim, windowSize, kernelFFT_1D, kernelMultConj, queue);

            


            debug_message("Finding correlation peak", DEBUG_LVL, 3, &currentTime);
            size_t localSize[2] = {windowSize,1};
            size_t numGroups[2] = {N_vec_cols, N_vec_rows};
            size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};
            cl_int2 outputDim;
            outputDim.x = N_vec_cols;
            outputDim.y = N_vec_rows;
            int idx=0;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &im1_windows); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_int2), &inputDim); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(int), &windowSize); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &U_GPU); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &V_GPU); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_int2), &outputDim); idx++;
            err=clEnqueueNDRangeKernel(queue, kernelMaxCorr, 2, NULL, globalSize, localSize,0, NULL, NULL);
            err = clFinish(queue);





            //----------------------------------------------------------------------------------------------------
            //-----------------------------------Saving Results---------------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Saving Results", DEBUG_LVL, 3, &currentTime);

            // Read the results from the device
            clEnqueueReadBuffer(queue, U_GPU, CL_TRUE, 0, vec_bytes, U, 0, NULL, NULL );
            clEnqueueReadBuffer(queue, V_GPU, CL_TRUE, 0, vec_bytes, V, 0, NULL, NULL );
            // convert the pixel displacements to velocity
            multiply_float_array_by_scalar(U, N_vec_rows*N_vec_cols, 1/dt);
            multiply_float_array_by_scalar(V, N_vec_rows*N_vec_cols, 1/dt);


            fprintf(fp, "Pass %d of %d\n", pass+1, N_pass);
            fprintf(fp, "Window size %d\n", windowSizes[pass]);
            fprintf(fp, "Rows %d\n", N_vec_rows);
            fprintf(fp, "Cols %d\n", N_vec_cols);
            fprintf(fp, "image_x,image_y,U,V\n");
            for(int i=0;i<N_vec_rows;i++){
                for(int j=0;j<N_vec_cols;j++){
                    fprintf(fp,"%d,%d,%.12f,%.12f\n",j*window_shift +1,i*window_shift +1,U[i*N_vec_cols + j],V[i*N_vec_cols + j]);
                }
            }
            fprintf(fp, "\n\n\n\n\n");



            //----------------------------------------------------------------------------------------------------
            //-----------------------------------Deallocating memory----------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Deallocating memory", DEBUG_LVL, 3, &currentTime);
            free(U);free(V);
            clReleaseMemObject(U_GPU);clReleaseMemObject(V_GPU);
            clReleaseMemObject(im1_windows);clReleaseMemObject(im2_windows);


        }
        fclose(fp);

    }
    clReleaseMemObject(im1_windows_max);clReleaseMemObject(im2_windows_max);
    clReleaseMemObject(im1_GPU);clReleaseMemObject(im2_GPU);
    clReleaseKernel(kernelFFT_1D);
    clReleaseKernel(kernelMultConj);
    clReleaseKernel(kernelMaxCorr);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseCommandQueue(queueNonBlocking);
    clReleaseContext(context);

    free(im1_filepath_template);free(im2_filepath_template);free(windowSizes);free(outputFile_template);
    
    debug_message("Program Complete", DEBUG_LVL, 0, &currentTime);
    return 0;
}
