#include "standardHeader.hpp"




const std::string kernelSource_tiffFunctions=R"(

__kernel void convert_to_float2(__global const uchar* input_data, __global float2* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        // Read the uint8_t data
        uchar val_uchar = input_data[gid];

        // Convert to a float
        float val_float = (float)val_uchar;

        // Create the cl_float2 vector with the s[0] element as the float value and s[1] as 0.0f
        float2 result_vec = (float2)(val_float, 0.0f);

        // Write the result to the output array
        output_data[gid] = result_vec;
    }
}


)";




/**
 * @brief Reads a single-channel TIFF file into a std::vector of the appropriate
 * unsigned integer type (uint8_t, uint16_t, or uint32_t).
 *
 * This function expects the TIFF file to be single-channel and contain
 * unsigned integer data. It determines the bit depth and returns the data
 * in a std::vector of the corresponding type within an ImageData struct.
 *
 * @param filePath The path to the TIFF file.
 * @return An ImageData struct containing the image dimensions and pixel data.
 * @throws std::runtime_error if the file cannot be opened, is not single-channel,
 * is not an unsigned integer format, or if reading fails.
 */
ImageData readTiffToAppropriateIntegerVector(const std::string& filePath) {
    // Open the TIFF file for reading
    TIFF* tif = TIFFOpen(filePath.c_str(), "r");
    if (!tif) {
        throw std::runtime_error("Error: Could not open TIFF file: " + filePath);
    }

    uint32_t width, height;
    uint16_t samplesPerPixel, bitsPerSample;//, sampleFormat;

    // Get image dimensions
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

    // Get number of samples per pixel (channels)
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    // Get bits per sample (e.g., 8, 16, 32)
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);

    // // Get sample format (e.g., unsigned int, signed int, IEEE float)
    // TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);
    // std::cout << sampleFormat << std::endl;

    // Validate image format: must be single channel
    if (samplesPerPixel != 1) {
        TIFFClose(tif);
        throw std::runtime_error("Error: Only single-channel TIFF files are supported by this function. Samples per pixel: " + std::to_string(samplesPerPixel));
    }

    // // Validate sample format: must be unsigned integer
    // if (sampleFormat != SAMPLEFORMAT_UINT) {
    //     TIFFClose(tif);
    //     throw std::runtime_error("Error: Unsupported sample format. This function only handles unsigned integer (UINT) data. Sample format: " + std::to_string(sampleFormat));
    // }

    ImageData imageData;
    imageData.width = width;
    imageData.height = height;
    imageData.dims.s[0]=width;
    imageData.dims.s[1]=height;
    size_t numPixels = static_cast<size_t>(width) * height;

    // Get the size of a single scanline in bytes
    tmsize_t scanlineSize = TIFFScanlineSize(tif);
    if (scanlineSize == 0) {
        TIFFClose(tif);
        throw std::runtime_error("Error: Could not determine scanline size.");
    }

    // Determine the appropriate type based on bitsPerSample and read data
    if (bitsPerSample == 8) {
        imageData.type = ImageData::DataType::UINT8;
        imageData.pixelBytes = sizeof(uint8_t);
        std::vector<uint8_t> data(numPixels);
        uint8_t* scanlineBuffer = static_cast<uint8_t*>(_TIFFmalloc(scanlineSize));
        if (!scanlineBuffer) {
            TIFFClose(tif);
            throw std::runtime_error("Error: Could not allocate scanline buffer for uint8_t.");
        }
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanlineBuffer, row) < 0) {
                _TIFFfree(scanlineBuffer);
                TIFFClose(tif);
                throw std::runtime_error("Error: Failed to read scanline " + std::to_string(row));
            }
            // Copy the data from the scanline buffer to the vector
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + (static_cast<size_t>(row) * width));
        }
        _TIFFfree(scanlineBuffer);
        imageData.pixelData = data; // Assign the vector to the variant
        imageData.sizeBytes = imageData.width * imageData.height * imageData.pixelBytes;
    } else if (bitsPerSample == 16) {
        imageData.type = ImageData::DataType::UINT16;
        imageData.pixelBytes = sizeof(uint16_t);
        std::vector<uint16_t> data(numPixels);
        uint16_t* scanlineBuffer = static_cast<uint16_t*>(_TIFFmalloc(scanlineSize));
        if (!scanlineBuffer) {
            TIFFClose(tif);
            throw std::runtime_error("Error: Could not allocate scanline buffer for uint16_t.");
        }
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanlineBuffer, row) < 0) {
                _TIFFfree(scanlineBuffer);
                TIFFClose(tif);
                throw std::runtime_error("Error: Failed to read scanline " + std::to_string(row));
            }
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + (static_cast<size_t>(row) * width));
        }
        _TIFFfree(scanlineBuffer);
        imageData.pixelData = data;
        imageData.sizeBytes = imageData.width * imageData.height * imageData.pixelBytes;
    } else if (bitsPerSample == 32) {
        imageData.type = ImageData::DataType::UINT32;
        imageData.pixelBytes = sizeof(uint32_t);
        std::vector<uint32_t> data(numPixels);
        uint32_t* scanlineBuffer = static_cast<uint32_t*>(_TIFFmalloc(scanlineSize));
        if (!scanlineBuffer) {
            TIFFClose(tif);
            throw std::runtime_error("Error: Could not allocate scanline buffer for uint32_t.");
        }
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanlineBuffer, row) < 0) {
                _TIFFfree(scanlineBuffer);
                TIFFClose(tif);
                throw std::runtime_error("Error: Failed to read scanline " + std::to_string(row));
            }
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + (static_cast<size_t>(row) * width));
        }
        _TIFFfree(scanlineBuffer);
        imageData.pixelData = data;
        imageData.sizeBytes = imageData.width * imageData.height * imageData.pixelBytes;

    } else {
        TIFFClose(tif);
        throw std::runtime_error("Error: Unsupported unsigned integer bits per sample: " + std::to_string(bitsPerSample));
    }

    // Close the TIFF file
    TIFFClose(tif);

    return imageData;
}


//__kernel void convert_to_float2(__global const uchar* input_data, __global float2* output_data, int N) {

cl_int uploadImage_and_convert_to_complex(ImageData& im, OpenCL_env& env, cl::Buffer& buffer, cl::Buffer& buffer_complex){
    cl_int err =CL_SUCCESS;

    // due to the unknown type of the data, this extra step is needed
    const void* host_ptr = std::visit([](const auto& vec) -> const void* {return vec.data();}, im.pixelData);
    if (!host_ptr) {
        // Handle error: host pointer is null
        return CL_INVALID_HOST_PTR; // A custom or predefined error code
    }
    try{err = env.queue.enqueueWriteBuffer( buffer, CL_TRUE, 0, im.sizeBytes, host_ptr);} catch (cl::Error& e) {std::cerr << "Error uploading to GPU" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}

    cl_int N = im.width*im.height;
    try {err = env.kernel_convert_im_to_complex.setArg(0, buffer);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_convert_im_to_complex.setArg(1, buffer_complex);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = env.kernel_convert_im_to_complex.setArg(2, sizeof(cl_int),&N);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(env.kernel_convert_im_to_complex, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_convert_im_to_complex" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}


