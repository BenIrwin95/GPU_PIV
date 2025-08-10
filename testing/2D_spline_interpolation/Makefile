# compiler
CC=g++

# compiler flags
CFLAGS = -Wall -g -std=c++20

# Directories to search for header files
INCLUDE_DIRS = -I../../src/ #-I/usr/include/alglib

# Directories to search for libraries
LDFLAGS = -L/usr/local/lib -L/usr/local/lib64 #-L/usr/lib64/

# Libraries to link against
LIBS = -ltiff -lm -lOpenCL -lfmt -lalglib

# C++ source files
SRCS = ./2D_spline_interpolation.cpp ../../src/OpenCL_utilities.cpp ../../src/inputFunctions.cpp ../../src/tiffFunctions.cpp ../../src/dataArrangement.cpp ../../src/Bspline_interpolation.cpp ../../src/FFT.cpp ../../src/complexMaths.cpp ../../src/determineCorrelation.cpp ../../src/vectorValidation.cpp ../../src/outputFunctions.cpp

# the object files
OBJS = $(SRCS:.cpp=.o)

# the created executable
TARGET = ./my_program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)

# Rule to compile .cpp files into object files
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
