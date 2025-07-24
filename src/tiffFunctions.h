#ifndef TIFF_FUNCTIONS_H
#define TIFF_FUNCTIONS_H
#include <stdint.h>   // Required for uint8_t, uint32_t, uint16_t
#include <tiffio.h>   // Required for libtiff functions


/*readSingleChannelTiff
 * function to read a single-channel (grayscale) tiff file at a specified location and pass its contents into an array
 * first output points towards the bottom left of the image
 * the last output points towards the top right of the image
 */
uint8_t* readSingleChannelTiff(const char* filePath, uint32_t* width, uint32_t* height);



// function to convert a uint8_t image into complex form
// this is needed for FFT
cl_float2* tiff2complex(uint8_t* input ,uint32_t width, uint32_t height);


int get_tiff_dimensions_single_channel(const char *filepath, uint32_t *width, uint32_t *height);
#endif
