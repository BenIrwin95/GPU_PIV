# Default compiler and flags
CC = g++
CFLAGS = -Wall -g -std=c++20

# OS-specific settings
ifeq ($(OS), Windows_NT)
    # Windows settings (e.g., using MSYS2 or MinGW)
    # LDFLAGS will point to where libraries are installed on Windows
    LDFLAGS = -L"C:/msys64/mingw64/lib"
    LIBS = -ltiff -lm -lOpenCL -lfmt
    TARGET = GPU_PIV.exe
else
    # Detect macOS vs Linux
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S), Darwin)
        # macOS settings
        CFLAGS += -Xpreprocessor -fopenmp
        LDFLAGS = -L/usr/local/opt/libomp/lib -L/usr/local/lib
        LIBS = -ltiff -lm -lOpenCL -lfmt -lomp
        TARGET = GPU_PIV
    else
        # Assume Linux (GNU/Linux)
        CFLAGS += -fopenmp
        LDFLAGS = -L/usr/local/lib -L/usr/local/lib64
        LIBS = -ltiff -lm -lOpenCL -lfmt
        TARGET = ./GPU_PIV
    endif
endif

# Common settings
SRCS = ./src/main.cpp ./src/OpenCL_utilities.cpp ./src/inputFunctions.cpp ./src/tiffFunctions.cpp ./src/dataArrangement.cpp ./src/bicubic_interpolation.cpp ./src/FFT.cpp ./src/complexMaths.cpp ./src/determineCorrelation.cpp ./src/vectorValidation.cpp ./src/outputFunctions.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
