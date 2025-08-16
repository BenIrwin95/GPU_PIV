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


__kernel void convert_float2_to_uint8(__global const float2* input_data, __global uchar* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        output_data[gid] = (uchar)input_data[gid].x;
    }
}




__kernel void convert_uint16_to_float2(__global const ushort* input_data, __global float2* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        // Read the uint8_t data
        ushort val_uchar = input_data[gid];

        // Convert to a float
        float val_float = (float)val_uchar;

        // Create the cl_float2 vector with the s[0] element as the float value and s[1] as 0.0f
        float2 result_vec = (float2)(val_float, 0.0f);

        // Write the result to the output array
        output_data[gid] = result_vec;
    }
}

__kernel void convert_float2_to_uint16(__global const float2* input_data, __global ushort* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        output_data[gid] = (ushort)input_data[gid].x;
    }
}



__kernel void convert_uint32_to_float2(__global const uint* input_data, __global float2* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        // Read the uint8_t data
        uint val_uchar = input_data[gid];

        // Convert to a float
        float val_float = (float)val_uchar;

        // Create the cl_float2 vector with the s[0] element as the float value and s[1] as 0.0f
        float2 result_vec = (float2)(val_float, 0.0f);

        // Write the result to the output array
        output_data[gid] = result_vec;
    }
}

__kernel void convert_float2_to_uint32(__global const float2* input_data, __global uint* output_data, int N) {
    int gid = get_global_id(0);

    if(gid<N){
        output_data[gid] = (uint)input_data[gid].x;
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
 
 Note: the first element of the output corresponds to the bottom left of the image
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
            size_t dest_row = height - 1 - row;
            size_t offset = dest_row * width;
            // Copy the data from the scanline buffer to the vector
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + offset);
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
            size_t dest_row = height - 1 - row;
            size_t offset = dest_row * width;
            // Copy the data from the scanline buffer to the vector
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + offset);
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
            size_t dest_row = height - 1 - row;
            size_t offset = dest_row * width;
            // Copy the data from the scanline buffer to the vector
            std::copy(scanlineBuffer, scanlineBuffer + width, data.begin() + offset);
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

    cl::Kernel correct_kernel;
    switch (im.type) {
        case ImageData::DataType::UINT8:
            correct_kernel = env.kernel_convert_im_to_complex;
            break;
        case ImageData::DataType::UINT16:
            correct_kernel = env.kernel_convert_im_to_complex_uint16;
            break;
        case ImageData::DataType::UINT32:
            correct_kernel = env.kernel_convert_im_to_complex_uint32;
            break;
        case ImageData::DataType::UNKNOWN:
        default:
            std::cout << "Data type is unknown or unsupported." << std::endl;
            return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }


    // due to the unknown type of the data, this extra step is needed
    const void* host_ptr = std::visit([](const auto& vec) -> const void* {return vec.data();}, im.pixelData);
    if (!host_ptr) {
        // Handle error: host pointer is null
        return CL_INVALID_HOST_PTR; // A custom or predefined error code
    }
    try{err = env.queue.enqueueWriteBuffer( buffer, CL_TRUE, 0, im.sizeBytes, host_ptr);} catch (cl::Error& e) {std::cerr << "Error uploading to GPU" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}

    cl_int N = im.width*im.height;
    try {err = correct_kernel.setArg(0, buffer);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 0" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = correct_kernel.setArg(1, buffer_complex);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 1" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}
    try {err = correct_kernel.setArg(2, sizeof(cl_int),&N);} catch (cl::Error& e) {std::cerr << "Error setting kernel argument 2" << std::endl;CHECK_CL_ERROR(e.err());return e.err();}


    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);
    try{
        env.queue.enqueueNDRangeKernel(correct_kernel, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_convert_im_to_complex" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    return err;
}







cl_int retrieveImageFromBuffer(cl::Buffer& im_buffer_complex, cl::Buffer& im_buffer, ImageData& im, OpenCL_env& env){
    cl_int err = CL_SUCCESS;

    // determine image type and select appropriate function to convert image from its float2 form to an integer
    cl::Kernel correct_kernel;
    switch (im.type) {
        case ImageData::DataType::UINT8:
            correct_kernel = env.kernel_convert_float2_to_uint8;
            break;
        case ImageData::DataType::UINT16:
            correct_kernel = env.kernel_convert_float2_to_uint16;
            break;
        case ImageData::DataType::UINT32:
            correct_kernel = env.kernel_convert_float2_to_uint32;
            break;
        case ImageData::DataType::UNKNOWN:
        default:
            std::cout << "Data type is unknown or unsupported." << std::endl;
            return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }
    cl_int N = im.width*im.height;
    try {err = correct_kernel.setArg(0, im_buffer_complex);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = correct_kernel.setArg(1, im_buffer);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}
    try {err = correct_kernel.setArg(2, sizeof(cl_int),&N);} catch (cl::Error& e) {CHECK_CL_ERROR(e.err());return e.err();}



    size_t N_local = 64;
    cl::NDRange local(N_local);
    size_t N_groups = ceil((float)N/N_local);
    cl::NDRange global(N_groups*N_local);

    try{
        env.queue.enqueueNDRangeKernel(correct_kernel, cl::NullRange, global, local);
    } catch (cl::Error& e) {
        std::cerr << "Error Enqueuing kernel_convert_im_to_complex" << std::endl;
        CHECK_CL_ERROR(e.err());
        return e.err();
    }
    env.queue.finish();

    // due to the unknown type of the data, this extra step is needed
    void* host_ptr = std::visit([](auto& vec) -> void* {return vec.data();}, im.pixelData);
    if (!host_ptr) {
        // Handle error: host pointer is null
        return CL_INVALID_HOST_PTR; // A custom or predefined error code
    }

    // after converting, pull image from GPU into the provided ImageData object
    env.queue.enqueueReadBuffer(im_buffer, CL_TRUE, 0, im.sizeBytes, host_ptr);



    return err;
}










/**
 * @brief Writes image data from an ImageData struct to a single-channel TIFF file.
 *
 * This function creates a new TIFF file at the specified path and writes the
 * pixel data contained within the ImageData struct. It supports 8-bit, 16-bit,
 * and 32-bit unsigned integer grayscale images. The function mirrors the
 * orientation logic of the provided read function, writing scanlines from
 * the bottom-up based on the read function's top-down orientation.
 *
 * @param imageData A constant reference to an ImageData struct containing the
 * image dimensions, data type, and pixel data.
 * @param filePath The path to the output TIFF file.
 * @throws std::runtime_error if the file cannot be created, if an unsupported
 * ImageData type is provided, or if any error occurs during writing.
 */
void writeTiffFromAppropriateIntegerVector(const ImageData& imageData, const std::string& filePath) {
    // Open the TIFF file for writing. "w" mode creates a new file or overwrites an existing one.
    TIFF* tif = TIFFOpen(filePath.c_str(), "w");
    if (!tif) {
        throw std::runtime_error("Error: Could not create TIFF file: " + filePath);
    }

    // --- Set essential TIFF tags ---
    // Image dimensions
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imageData.width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imageData.height);

    // Number of samples per pixel.
    // Based on your read function, this utility assumes single-channel images.
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);

    // Planar configuration: how pixel data is stored.
    // PLANARCONFIG_CONTIG means samples for a pixel are stored contiguously (e.g., RGBRGB).
    // For single-channel, this is the standard.
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    // Photometric interpretation: how pixel values relate to color.
    // PHOTOMETRIC_MINISBLACK means 0 is black, increasing values are lighter (standard for grayscale).
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    // Orientation of the image. ORIENTATION_TOPLEFT is standard (row 0 is the top).
    // The read function reverses rows, so we'll ensure consistency here.
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

    // Determine and set bits per sample based on the ImageData's DataType.
    uint16_t bitsPerSample = 0;
    if (imageData.type == ImageData::DataType::UINT8) {
        bitsPerSample = 8;
    } else if (imageData.type == ImageData::DataType::UINT16) {
        bitsPerSample = 16;
    } else if (imageData.type == ImageData::DataType::UINT32) {
        bitsPerSample = 32;
    } else {
        TIFFClose(tif);
        throw std::runtime_error("Error: Unsupported ImageData type for writing. Only UINT8, UINT16, UINT32 are supported.");
    }
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);

    // Sample format: specifies if data is unsigned int, signed int, or float.
    // Your read function explicitly validates for UINT, so we set it here.
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);

    // Compression method. LZW is a good lossless compression option.
    // Use COMPRESSION_NONE for no compression.
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    // Rows per strip: how many rows are grouped into a single strip of data.
    // TIFFDefaultStripSize is a good way to let libtiff decide an optimal value.
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));

    // Get the expected size of a single scanline in bytes. This is important for buffer handling.
    tmsize_t scanlineSize = TIFFScanlineSize(tif);
    if (scanlineSize == 0) {
        TIFFClose(tif);
        throw std::runtime_error("Error: Could not determine scanline size for writing.");
    }

    // --- Write pixel data scanline by scanline ---
    try {
        // Use std::visit or if-else with std::get to access the correct vector type
        // from the std::variant.
        if (imageData.type == ImageData::DataType::UINT8) {
            // Get a reference to the uint8_t vector from the variant.
            const auto& data = std::get<std::vector<uint8_t>>(imageData.pixelData);
            for (uint32_t row = 0; row < imageData.height; ++row) {
                // Your read function copies to `dest_row = height - 1 - row`.
                // To maintain consistency and write the image correctly from your in-memory representation,
                // we need to reverse this for writing. So, the data for 'row' will come from
                // the 'height - 1 - row' in your pixelData vector.
                size_t src_row_index = imageData.height - 1 - row;
                const uint8_t* scanlineBuffer = data.data() + (src_row_index * imageData.width);

                // TIFFWriteScanline writes a single scanline (row) of image data.
                // The const_cast is necessary because TIFFWriteScanline expects a void*
                // even though it doesn't modify the buffer for writing.
                if (TIFFWriteScanline(tif, const_cast<void*>(static_cast<const void*>(scanlineBuffer)), row) < 0) {
                    throw std::runtime_error("Error: Failed to write scanline " + std::to_string(row) + " for UINT8 data.");
                }
            }
        } else if (imageData.type == ImageData::DataType::UINT16) {
            const auto& data = std::get<std::vector<uint16_t>>(imageData.pixelData);
            for (uint32_t row = 0; row < imageData.height; ++row) {
                size_t src_row_index = imageData.height - 1 - row;
                const uint16_t* scanlineBuffer = data.data() + (src_row_index * imageData.width);
                if (TIFFWriteScanline(tif, const_cast<void*>(static_cast<const void*>(scanlineBuffer)), row) < 0) {
                    throw std::runtime_error("Error: Failed to write scanline " + std::to_string(row) + " for UINT16 data.");
                }
            }
        } else if (imageData.type == ImageData::DataType::UINT32) {
            const auto& data = std::get<std::vector<uint32_t>>(imageData.pixelData);
            for (uint32_t row = 0; row < imageData.height; ++row) {
                size_t src_row_index = imageData.height - 1 - row;
                const uint32_t* scanlineBuffer = data.data() + (src_row_index * imageData.width);
                if (TIFFWriteScanline(tif, const_cast<void*>(static_cast<const void*>(scanlineBuffer)), row) < 0) {
                    throw std::runtime_error("Error: Failed to write scanline " + std::to_string(row) + " for UINT32 data.");
                }
            }
        }
    } catch (const std::bad_variant_access& e) {
        // This exception is thrown if std::get attempts to access a type not currently held by the variant.
        // It indicates a mismatch between imageData.type and the actual content of imageData.pixelData.
        TIFFClose(tif);
        throw std::runtime_error("Error: Mismatch between ImageData::type and actual pixelData variant content: " + std::string(e.what()));
    } catch (...) {
        // Catch any other unexpected exceptions during the writing process.
        TIFFClose(tif);
        throw; // Re-throw the caught exception.
    }

    // --- Finalize and close the TIFF file ---
    TIFFClose(tif);
}

