#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include "standardLibraries.hpp"
#include "globalVars.hpp"

//--------------------------------------------------------------------------------
//-------------------------------inputFunctions---------------------------------
//--------------------------------------------------------------------------------

int findIntegerAfterKeyword(const std::string& filename, const std::string& keyword);
std::string findRestOfLineAfterKeyword(const std::string& filename, const std::string& keyword);
std::vector<int> findIntegersAfterKeyword(const std::string& filename, const std::string& keyword);

//--------------------------------------------------------------------------------
//-------------------------------OpenCL_utilities---------------------------------
//--------------------------------------------------------------------------------
const char* get_cl_error_string(cl_int err);
void print_cl_error(cl_int err, const std::string& filename, int line_number);



//--------------------------------------------------------------------------------
//-------------------------------tiffFunctions------------------------------------
//--------------------------------------------------------------------------------

ImageData readTiffToAppropriateIntegerVector(const std::string& filePath);

#endif
