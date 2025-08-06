#ifndef MACROS_H
#define MACROS_H

#define ERROR_MSG_OPENCL(err) fprintf(stderr,"ERROR: %s - %s:%d:%s:\n", getOpenCLErrorString(err), __FILE__, __LINE__, __func__)

#endif
