# Makefile for Dynamic Hypergraph Analysis Project
# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++11 -O2
CXX_FLAGS = -std=c++11 -O2 -Wall

# Target executable
TARGET = main

# Source files
CUDA_SOURCES = main.cu
CPP_SOURCES = graphGeneration.cpp
HEADERS = graphGeneration.hpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Compile CUDA source files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Clean and rebuild
rebuild: clean all

# Install dependencies (if needed)
install:
	@echo "Make sure CUDA toolkit is installed and nvcc is in PATH"

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the project (default)"
	@echo "  run      - Build and run the program"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and rebuild"
	@echo "  install  - Show installation instructions"
	@echo "  help     - Show this help message"

# Phony targets
.PHONY: all run clean rebuild install help
