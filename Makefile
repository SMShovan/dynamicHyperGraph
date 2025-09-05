# Makefile for Dynamic Hypergraph Analysis Project
# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++11 -O2
CXX_FLAGS = -std=c++11 -O2 -Wall

# Directories
SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files
CUDA_SOURCES = $(SRC_DIR)/main.cu
CPP_SOURCES = $(SRC_DIR)/graphGeneration.cpp
HEADERS = $(SRC_DIR)/graphGeneration.hpp

# Object files
CUDA_OBJECTS = $(BUILD_DIR)/main.o
CPP_OBJECTS = $(BUILD_DIR)/graphGeneration.o

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

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
