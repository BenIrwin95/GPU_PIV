// standard libraries
#include "standardLibraries.h"
#include "macros.h"
#include "functions.h"
#include "globalVars.h"

/*readSingleChannelTiff
 * function to read a single-channel (grayscale) tiff file at a specified location and pass its contents into an array
 * first output points towards the bottom left of the image
 * the last output points towards the top right of the image
 */
uint8_t* readSingleChannelTiff(const char* filePath, uint32_t* width, uint32_t* height) {
    // Open the TIFF file in read mode ("r")
    TIFF* tif = TIFFOpen(filePath, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file '%s'.\n", filePath);
        return NULL;
    }

    uint32_t w, h;
    uint16_t samplesPerPixel, bitsPerSample;

    // Retrieve image dimensions (width and height) from TIFF tags
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

    // Retrieve the number of samples per pixel (e.g., 1 for grayscale, 3 for RGB)
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    // Retrieve the number of bits per sample (e.g., 8 for 8-bit, 16 for 16-bit)
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);

    // --- Validation Checks ---

    // 1. Check if the image is single-channel
    if (samplesPerPixel != 1) {
        fprintf(stderr, "Error: TIFF file '%s' is not a single-channel image (SamplesPerPixel = %hu).\n", filePath, samplesPerPixel);
        TIFFClose(tif);
        return NULL;
    }

    // 2. Check if the bit depth is 8-bit (uint8_t)
    if (bitsPerSample != 8) {
        fprintf(stderr, "Error: TIFF file '%s' has unsupported BitsPerSample = %hu. This function only supports 8-bit single-channel images.\n", filePath, bitsPerSample);
        // To support 16-bit, you would change `uint8_t* pixels` to `uint16_t* pixels`
        // and adjust `sizeof(uint8_t)` to `sizeof(uint16_t)` and `memcpy` accordingly.
        TIFFClose(tif);
        return NULL;
    }

    // --- Memory Allocation ---

    // Allocate memory for the entire image's pixel data.
    // For an 8-bit single-channel image, each pixel takes 1 byte.
    uint8_t* pixels = (uint8_t*)malloc((size_t)w * h * sizeof(uint8_t));
    if (!pixels) {
        fprintf(stderr, "Error: Could not allocate memory for pixel data.\n");
        TIFFClose(tif);
        return NULL;
    }

    // Allocate a buffer for reading one scanline at a time.
    // TIFFScanlineSize(tif) returns the size in bytes of a single scanline.
    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
    if (!buf) {
        fprintf(stderr, "Error: Could not allocate scanline buffer.\n");
        free(pixels); // Free previously allocated pixel memory
        TIFFClose(tif);
        return NULL;
    }

    // --- Reading Pixel Data ---

    // Loop through each scanline (row) of the image
    // To make pixels[0] refer to the bottom-left, we copy the top scanline
    // to the last row of our `pixels` array, and the bottom scanline to the first row.
    for (uint32_t row = 0; row < h; row++) {
        // Read the current scanline into the buffer `buf`
        if (TIFFReadScanline(tif, buf, row, 0) < 0) {
            fprintf(stderr, "Error: Failed to read scanline %u from TIFF file '%s'.\n", row, filePath);
            _TIFFfree(buf);   // Free scanline buffer
            free(pixels);     // Free pixel data
            TIFFClose(tif);   // Close TIFF file
            return NULL;
        }
        // Copy the data from the scanline buffer to our main pixel array.
        // The destination is `pixels` starting at the offset for the current row.
        // The size to copy is `w * sizeof(uint8_t)` bytes (width pixels * 1 byte/pixel).
        // To reverse the vertical order, we calculate the target row in `pixels` as (h - 1 - row).
        // This way pixels[0] refers to the bottom left corner
        memcpy(&pixels[(h - 1 - row) * w], buf, (size_t)w * sizeof(uint8_t));
    }

    // --- Cleanup ---

    // Free the temporary scanline buffer
    _TIFFfree(buf);
    // Close the TIFF file
    TIFFClose(tif);

    //printf("Image loaded: %s \t (%d x %d) pixels\n", filePath, w, h);

    // Pass the image dimensions back to the caller via the pointers
    *width = w;
    *height = h;

    return pixels; // Return the pointer to the pixel data
}


int get_tiff_dimensions_single_channel(const char *filepath, uint32_t *width, uint32_t *height) {
    TIFF *tif = NULL;
    uint16_t samplesperpixel = 0;
    uint16_t photometric = 0;

    // Initialize output parameters to 0
    if (width) *width = 0;
    if (height) *height = 0;

    // Open the TIFF file
    tif = TIFFOpen(filepath, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file: %s\n", filepath);
        return -1; // File not found or not a valid TIFF
    }

    // Get image width and height
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width)) {
        fprintf(stderr, "Error: Could not retrieve image width from %s\n", filepath);
        TIFFClose(tif);
        return -2;
    }
    if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height)) {
        fprintf(stderr, "Error: Could not retrieve image height from %s\n", filepath);
        TIFFClose(tif);
        return -2;
    }

    // Check for samples per pixel (should be 1 for single channel)
    if (!TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel)) {
        fprintf(stderr, "Error: Could not retrieve SamplesPerPixel from %s\n", filepath);
        TIFFClose(tif);
        return -3;
    }
    if (samplesperpixel != 1) {
        fprintf(stderr, "Error: Image is not single-channel. SamplesPerPixel = %hu in %s\n", samplesperpixel, filepath);
        TIFFClose(tif);
        return -3;
    }

    // Check photometric interpretation for single-channel (grayscale)
    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric)) {
        fprintf(stderr, "Error: Could not retrieve PhotometricInterpretation from %s\n", filepath);
        TIFFClose(tif);
        return -4;
    }
    if (photometric != PHOTOMETRIC_MINISBLACK && photometric != PHOTOMETRIC_MINISWHITE) {
        fprintf(stderr, "Error: Single-channel image has unsupported photometric interpretation (%hu) in %s. Expected MINISBLACK or MINISWHITE.\n", photometric, filepath);
        TIFFClose(tif);
        return -4;
    }

    // Close the TIFF file
    TIFFClose(tif);

    return 0; // Success
}





cl_float2* tiff2complex(const uint8_t* input ,uint32_t width, uint32_t height){
    cl_float2* output = (cl_float2*)malloc(width*height*sizeof(cl_float2));
    for(int i=0;i<height*width;i++){
        output[i].x = input[i];
        output[i].y = 0.0;
    }
    return output;
}

