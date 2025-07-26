// standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
// custom headers
#include "OpenCL_utilities.h"
#include "tiffFunctions.h" // imports functions for loading and saving tiff images
#include "utilities.h" // imports miscellaneous functions
#include "inputFunctions.h" // imports functions for taking inputs from a text file
#include "determineCorrelation.h" // applies the GPU_FFT library to calcualte correlation
#include "dataArrangement.h" // for arranging the data in a convenient arrangement for FFT
#include "gridFunctions.h" // for creating X and Y and interpolating between grids of different sizes
#include "vectorValidation.h" // for validating and replacing bad vectors
#include "libGPU_FFT.h"



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
    int status = 0;
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

    if(status == 1){
        printf("ERROR: Something went wrong loading the input file\n");
        free(im1_filepath_template);free(im2_filepath_template);free(windowSizes);free(outputFile_template);
        return 1;
    }



    //----------------------------------------------------------------------------------------------------
    //-----------------------------------OpenCL initialisation--------------------------------------------
    //----------------------------------------------------------------------------------------------------
    debug_message("Initial OpenCL setup", DEBUG_LVL, 0, &currentTime);
    cl_platform_id platform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_command_queue queueNonBlocking;           // command queue
    cl_program program;               // program
    cl_kernel kernelMultConj;
    cl_kernel kernelMaxCorr;
    cl_kernel kernelFFT_1D;
    cl_kernel kernel_uniformTiling;
    cl_kernel kernel_offsetTiling;
    cl_kernel kernel_identifyInvalidVectors;
    cl_kernel kernel_correctInvalidVectors;
    cl_int err;
    err = initialise_OpenCL(&platform, &device_id, &context, &queue, &queueNonBlocking, &program, &kernelFFT_1D, &kernelMultConj, &kernelMaxCorr, &kernel_uniformTiling, &kernel_offsetTiling, &kernel_identifyInvalidVectors, &kernel_correctInvalidVectors);
    if(err!=CL_SUCCESS){
      return 1;
    }
    
    // initialise buffers
    debug_message("Initialise buffers", DEBUG_LVL, 0, &currentTime);

    // initialise pointers to hold pointers to each of the arrays generated each pass
    // need so we can interpolate U and V between passes
    float** X_passes = (float**)malloc(N_pass * sizeof(float*));
    float** Y_passes = (float**)malloc(N_pass * sizeof(float*));
    float** U_passes = (float**)malloc(N_pass * sizeof(float*));
    float** V_passes = (float**)malloc(N_pass * sizeof(float*));
    cl_int2* vecDim_passes = (cl_int2*)malloc(N_pass * sizeof(cl_int2)); // the dimensions of the arrays in each pass
    
    cl_mem* X_GPU_passes = (cl_mem*)malloc(N_pass*sizeof(cl_mem));
    cl_mem* Y_GPU_passes = (cl_mem*)malloc(N_pass*sizeof(cl_mem));
    cl_mem* U_GPU_passes = (cl_mem*)malloc(N_pass*sizeof(cl_mem));
    cl_mem* V_GPU_passes = (cl_mem*)malloc(N_pass*sizeof(cl_mem));



    // determining the max size of data structures to prevent repeated malloc's and free's
    size_t maxTiledInputSize = 0;
    size_t flagsSize_max=0;
    // retrieve size of 1 image
    char temp_file[MAX_FILEPATH_LENGTH];
    snprintf(temp_file, sizeof(temp_file),im1_filepath_template, im1_frame_start);
    uint32_t temp_width, temp_height;
    get_tiff_dimensions_single_channel(temp_file, &temp_width, &temp_height);
    for(int i=0;i<N_pass;i++){
        int windowSize = windowSizes[i];
        double overlap = 0.5;
        int window_shift = (1.0-overlap)*windowSize;
        cl_int2 vecDim;
        vecDim.x = floor((temp_width-windowSize)/window_shift);
        vecDim.y = floor((temp_height-windowSize)/window_shift);

        size_t bytesNeeded = (vecDim.x*windowSize)*(vecDim.y*windowSize) * sizeof(cl_float2);
        if(bytesNeeded > maxTiledInputSize){
            maxTiledInputSize = bytesNeeded;
        }

        // allocate space for X,Y,U,V
        size_t vecSize = vecDim.x*vecDim.y*sizeof(float);
        size_t flagsSize = vecDim.x*vecDim.y*sizeof(int);
        if(flagsSize>flagsSize_max){flagsSize_max=flagsSize;}
        X_passes[i] = (float*)malloc(vecSize);
        Y_passes[i] = (float*)malloc(vecSize);
        U_passes[i] = (float*)malloc(vecSize);
        V_passes[i] = (float*)malloc(vecSize);
        vecDim_passes[i]=vecDim;

        // populate X and Y
        populateGrid(X_passes[i], Y_passes[i], vecDim, windowSize, window_shift);
        
        
        //allocate space for U and V on GPU
        X_GPU_passes[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize, NULL, &err);
        Y_GPU_passes[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize, NULL, &err);
        U_GPU_passes[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize, NULL, &err);
        V_GPU_passes[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize, NULL, &err);

    }


    // create a buffer for flags (needed for vector validation)
    cl_mem flags_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, flagsSize_max, NULL, &err);

    // image buffers
    cl_int2 imageDim;
    imageDim.x=temp_width;imageDim.y=temp_height;
    size_t imageBytes = imageDim.x*imageDim.y*sizeof(cl_float2);
    cl_mem im1_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, imageBytes, NULL, &err); // we assume our image size will be constant and thus assign a buffer in advance for frames
    cl_mem im2_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, imageBytes, NULL, &err);
    cl_mem im1_windows = clCreateBuffer(context, CL_MEM_READ_WRITE, maxTiledInputSize, NULL, &err); // the buffer that will hold all the windowed parts of the image during a pass
    cl_mem im2_windows = clCreateBuffer(context, CL_MEM_READ_WRITE, maxTiledInputSize, NULL, &err); // allocate space based on the max expected size, during each pass we just use as much as we need



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
        err = clEnqueueWriteBuffer( queue, im1_GPU, CL_TRUE, 0, imageBytes, im1, 0, NULL, NULL );
        err = clEnqueueWriteBuffer( queue, im2_GPU, CL_TRUE, 0, imageBytes, im2, 0, NULL, NULL );
        free(im1);free(im2);



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
            double overlap = 0.5;
            int window_shift = (1.0-overlap)*windowSize;
            float dt = 1.0;
            
            float* X = X_passes[pass];
            float* Y = Y_passes[pass];
            float* U = U_passes[pass];
            float* V = V_passes[pass];
            cl_int2 vecDim = vecDim_passes[pass];

            // populate U and V with last pass
            if(pass>0){
                gridInterpolate(X_passes[pass-1], Y_passes[pass-1],U_passes[pass-1], V_passes[pass-1], vecDim_passes[pass-1], X, Y,U, V, vecDim);
                round_float_array(U, vecDim.x*vecDim.y);
                round_float_array(V, vecDim.x*vecDim.y);
            }
            
            // retrieve previously allocated memory on GPU for U and V
            cl_mem X_GPU = X_GPU_passes[pass];
            cl_mem Y_GPU = Y_GPU_passes[pass];
            cl_mem U_GPU = U_GPU_passes[pass];
            cl_mem V_GPU = V_GPU_passes[pass];
            size_t vec_bytes = vecDim.x*vecDim.y*sizeof(float);

            err = clEnqueueWriteBuffer( queue, X_GPU, CL_TRUE, 0, vec_bytes, X, 0, NULL, NULL );
            err = clEnqueueWriteBuffer( queue, Y_GPU, CL_TRUE, 0, vec_bytes, Y, 0, NULL, NULL );


            // fill with zeros
            //float zero_float_pattern = 0.0f;
            if(pass==0){
                float zero_float_pattern = 0.0f;
                size_t pattern_size = sizeof(float);
                err = clEnqueueFillBuffer(queue, U_GPU, &zero_float_pattern, pattern_size, 0, vec_bytes, 0, NULL, NULL);
                err = clEnqueueFillBuffer(queue, V_GPU, &zero_float_pattern, pattern_size, 0, vec_bytes, 0, NULL, NULL);
            } else {
                err = clEnqueueWriteBuffer( queue, U_GPU, CL_TRUE, 0, vec_bytes, U, 0, NULL, NULL );
                err = clEnqueueWriteBuffer( queue, V_GPU, CL_TRUE, 0, vec_bytes, V, 0, NULL, NULL );
            }

            //----------------------------------------------------------------------------------------------------
            //-------------------------------Initialising windows-------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Initialising windows", DEBUG_LVL, 3, &currentTime);

            debug_message("Copying windows to tiles", DEBUG_LVL, 4, &currentTime);
            uniformly_tile_data(im1_GPU, imageDim, windowSize, window_shift, vecDim, im1_windows, kernel_uniformTiling, queue);
            offset_tile_data(im2_GPU, imageDim, windowSize, window_shift, U_GPU, V_GPU, vecDim, im2_windows, kernel_offsetTiling, queue);
            err = clFinish(queue);

            
            // find correlations
            debug_message("Calculating correlation", DEBUG_LVL, 3, &currentTime);
            cl_int2 inputDim;
            inputDim.x = vecDim.x*windowSize;
            inputDim.y = vecDim.y*windowSize;
            FFT_corr_tiled (im1_windows,im2_windows, inputDim, windowSize, kernelFFT_1D, kernelMultConj, queue);

            


            debug_message("Finding correlation peak", DEBUG_LVL, 3, &currentTime);
            size_t localSize[2] = {windowSize,1};
            size_t numGroups[2] = {vecDim.x, vecDim.y};
            size_t globalSize[2] = {numGroups[0]*localSize[0],numGroups[1]*localSize[1]};
            int idx=0;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &im1_windows); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_int2), &inputDim); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(int), &windowSize); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &U_GPU); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_mem), &V_GPU); idx++;
            err = clSetKernelArg(kernelMaxCorr, idx, sizeof(cl_int2), &vecDim); idx++;
            err=clEnqueueNDRangeKernel(queue, kernelMaxCorr, 2, NULL, globalSize, localSize,0, NULL, NULL);
            err = clFinish(queue);

          
            debug_message("Validating vectors", DEBUG_LVL, 4, &currentTime);
            //validateVectors(X,Y, U, V, vecDim);
            //identifyInvalidVectors(U_GPU, V_GPU, flags_GPU, vecDim, kernel_identifyInvalidVectors, queue);
            validateVectors(X_GPU, Y_GPU, U_GPU, V_GPU, flags_GPU, vecDim, kernel_identifyInvalidVectors, kernel_correctInvalidVectors, queue);
            //----------------------------------------------------------------------------------------------------
            //-----------------------------------Saving Results---------------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Saving Results", DEBUG_LVL, 3, &currentTime);

            // Read the results from the device
            clEnqueueReadBuffer(queue, U_GPU, CL_TRUE, 0, vec_bytes, U, 0, NULL, NULL );
            clEnqueueReadBuffer(queue, V_GPU, CL_TRUE, 0, vec_bytes, V, 0, NULL, NULL );

            // convert the pixel displacements to velocity
            multiply_float_array_by_scalar(U, vecDim.x*vecDim.y, 1/dt);
            multiply_float_array_by_scalar(V, vecDim.x*vecDim.y, 1/dt);


            fprintf(fp, "Pass %d of %d\n", pass+1, N_pass);
            fprintf(fp, "Window size %d\n", windowSizes[pass]);
            fprintf(fp, "Rows %d\n", vecDim.y);
            fprintf(fp, "Cols %d\n", vecDim.x);
            fprintf(fp, "image_x,image_y,U,V\n");
            for(int i=0;i<vecDim.y;i++){
                for(int j=0;j<vecDim.x;j++){
                    fprintf(fp,"%.2f,%.2f,%.12f,%.12f\n",X[i*vecDim.x + j],Y[i*vecDim.x + j],U[i*vecDim.x + j],V[i*vecDim.x + j]);
                }
            }
            fprintf(fp, "\n\n\n\n\n");



            //----------------------------------------------------------------------------------------------------
            //-----------------------------------Deallocating memory----------------------------------------------
            //----------------------------------------------------------------------------------------------------
            debug_message("Deallocating memory", DEBUG_LVL, 3, &currentTime);
            


        }
        fclose(fp);

    }
    

    debug_message("Cleaning up", DEBUG_LVL, 3, &currentTime);
    
    for(int i=0;i<N_pass;i++){
        free(X_passes[i]);
        free(Y_passes[i]);
        free(U_passes[i]);
        free(V_passes[i]);
        clReleaseMemObject(X_GPU_passes[i]);
        clReleaseMemObject(Y_GPU_passes[i]);
        clReleaseMemObject(U_GPU_passes[i]);
        clReleaseMemObject(V_GPU_passes[i]);
    }
    free(X_passes);free(Y_passes);free(U_passes);free(V_passes);free(vecDim_passes);
    free(X_GPU_passes);free(Y_GPU_passes);free(U_GPU_passes);free(V_GPU_passes);

    
    clReleaseMemObject(im1_GPU);clReleaseMemObject(im2_GPU);
    clReleaseMemObject(im1_windows);clReleaseMemObject(im2_windows);
    clReleaseMemObject(flags_GPU);
    clReleaseKernel(kernelFFT_1D);
    clReleaseKernel(kernelMultConj);
    clReleaseKernel(kernelMaxCorr);
    clReleaseKernel(kernel_uniformTiling);
    clReleaseKernel(kernel_offsetTiling);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseCommandQueue(queueNonBlocking);
    clReleaseContext(context);

    free(im1_filepath_template);free(im2_filepath_template);free(windowSizes);free(outputFile_template);
    
    debug_message("Program Complete", DEBUG_LVL, 0, &currentTime);
    return 0;
}
