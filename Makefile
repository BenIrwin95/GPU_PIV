# compiler
CC=gcc

# compiler flags
# -g		adds debugging information
# -Wall		turns on compiler warnings
CFLAGS = -Wall -g

# Directories to search for header files (using -I)
INCLUDE_DIRS = -I./GPU_FFT
# Directories to search for libraries (using -L)
LDFLAGS = -L/usr/local/lib -L/usr/local/lib64  -L./GPU_FFT
# -L./usr/local/cuda-12.5/targets/x86_64-linux/lib/

# Libraries to link against (using -l)
LIBS = -ltiff -lm -lOpenCL -lGPU_FFT


# src files
SRCS = ./src/main.c ./src/utilities.c ./src/inputFunctions.c ./src/FFT_functions.c ./src/mainKernel.c
# the object files the .c files get converted into
OBJS = $(SRCS:.c=.o)

# the created executable
TARGET = .my_program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET) # link obj files into a final executable

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@ # compile .c files into obj files 

clean:
	rm -f $(OBJS) $(TARGET) # if doing "make clean": remove the obj files once done
