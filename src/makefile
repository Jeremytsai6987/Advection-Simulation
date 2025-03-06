# Makefile for compiling the simulation code

# Variables
CC = mpicc
CFLAGS = -fopenmp -O2 -march=native -funsafe-math-optimizations
TARGET = my_advection
SOURCES = main.c simulation.c nrutil.c
LDFLAGS = -lm

# Default target
all: $(TARGET)

# Compile target
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET) *.o
