#include "standardHeader.hpp"



const char* get_cl_error_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        //case CL_INVALID_WORK_ITEM_LIMIT:            return "CL_INVALID_WORK_ITEM_LIMIT";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        //case CL_INVALID_NUM_DEVICES:                return "CL_INVALID_NUM_DEVICES";
        //case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE"; // Duplicate, but common
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";
        // OpenCL 2.x and 3.x errors
        case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";
        case CL_INVALID_SPEC_ID:                    return "CL_INVALID_SPEC_ID";
        //case CL_MAX_STATIC_CONST_SIZE_EXCEEDED:     return "CL_MAX_STATIC_CONST_SIZE_EXCEEDED";

        default:                                    return "UNKNOWN OPENCL ERROR";
    }
}


void print_cl_error(cl_int err, const std::string& filename, int line_number) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: " << get_cl_error_string(err)
        << " at " << filename << ":" << line_number << std::endl;
        // Optionally, you can throw an exception here
    }
}



cl_int inititialise_OpenCL_buffers(OpenCL_env& env, PIVdata& piv_data, ImageData& im){
    cl_int err = CL_SUCCESS;
    uint32_t maxArrLen=0;
    cl_int2 maxArrSize;
    uint32_t max_ImWindowed_Len=0;
    for(int i=0;i<piv_data.N_pass;i++){
        uint32_t arrLen = piv_data.arrSize[i].s[0]*piv_data.arrSize[i].s[1];
        if(arrLen > maxArrLen){
            maxArrLen = arrLen;
            maxArrSize = piv_data.arrSize[i];
        }
        uint32_t ImWindowed_Len = arrLen * (piv_data.window_sizes[i]*piv_data.window_sizes[i]);
        if(ImWindowed_Len > max_ImWindowed_Len){
            max_ImWindowed_Len = ImWindowed_Len;
        }
    }
    env.im1 = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*im.pixelBytes, NULL, &err); if(err != CL_SUCCESS){return err;}
    env.im2 = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*im.pixelBytes, NULL, &err); if(err != CL_SUCCESS){return err;}
    env.im1_complex = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.im2_complex = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.im1_windows = cl::Buffer(env.context, CL_MEM_READ_WRITE, max_ImWindowed_Len*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.im2_windows = cl::Buffer(env.context, CL_MEM_READ_WRITE, max_ImWindowed_Len*sizeof(cl_float2), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.X = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.Y = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.U = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.V = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.flags = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(int), NULL, &err); if(err != CL_SUCCESS){return err;}

    env.x_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrSize.s[0]*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.y_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrSize.s[1]*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.U_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.V_ref = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrLen*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.x_vals = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrSize.s[0]*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.y_vals = cl::Buffer(env.context, CL_MEM_READ_WRITE, maxArrSize.s[1]*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}

    env.imageShifts_x = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.imageShifts_y = cl::Buffer(env.context, CL_MEM_READ_WRITE, im.width*im.height*sizeof(cl_float), NULL, &err); if(err != CL_SUCCESS){return err;}

    // some useful stuff for image interpolation that we will never need to change
    env.x_vals_im = cl::Buffer(env.context, CL_MEM_READ_ONLY, im.width*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    env.y_vals_im = cl::Buffer(env.context, CL_MEM_READ_ONLY, im.height*sizeof(float), NULL, &err); if(err != CL_SUCCESS){return err;}
    std::vector<float> im_idx_x(im.width);
    std::vector<float> im_idx_y(im.height);
    for(unsigned int i=0;i<im.width;i++){
        im_idx_x[i] = i;
    }
    for(unsigned int i=0;i<im.height;i++){
        im_idx_y[i] = i;
    }
    env.queue.enqueueWriteBuffer( env.x_vals_im, CL_TRUE, 0, im_idx_x.size()*sizeof(float), im_idx_x.data());
    env.queue.enqueueWriteBuffer( env.y_vals_im, CL_TRUE, 0, im_idx_y.size()*sizeof(float), im_idx_y.data());

    return err;
}

