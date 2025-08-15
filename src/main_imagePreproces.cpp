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


    ImageData im = readTiffToAppropriateIntegerVector("./example_resources/cam1_im_000_A.tiff");

    writeTiffFromAppropriateIntegerVector(im, "./example_resources/test.tiff");

    return 0;
}
