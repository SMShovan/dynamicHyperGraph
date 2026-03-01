# Makefile for Dynamic Hypergraph Analysis Project
# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++17 -O2 -arch=sm_86
CXX_FLAGS = -std=c++17 -O2 -Wall

# Directories
SRC_DIR = src
UTILS_DIR = utils
INCLUDE_DIR = include
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main
TARGET_TYPE1 = $(BUILD_DIR)/type1
TARGET_TYPE2 = $(BUILD_DIR)/type2
TARGET_TYPE3 = $(BUILD_DIR)/type3

# Source files
STRUCT_DIR = structure
KERNEL_DIR = kernel
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/HMotifCount.cu $(SRC_DIR)/HMotifCountUpdate.cu $(STRUCT_DIR)/operations.cu \
               $(KERNEL_DIR)/insert_reuse.cu $(KERNEL_DIR)/unfill.cu \
               $(KERNEL_DIR)/payload.cu $(KERNEL_DIR)/build_tree.cu \
               $(KERNEL_DIR)/delete_avail.cu $(KERNEL_DIR)/find.cu
CPP_SOURCES = $(SRC_DIR)/graphGeneration.cpp $(UTILS_DIR)/utils.cpp $(UTILS_DIR)/printUtils.cpp $(UTILS_DIR)/flatten.cpp
HEADERS = $(INCLUDE_DIR)/graphGeneration.hpp $(INCLUDE_DIR)/utils.hpp $(INCLUDE_DIR)/printUtils.hpp

# Object files (main entry point)
CUDA_OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/HMotifCount.o $(BUILD_DIR)/HMotifCountUpdate.o $(BUILD_DIR)/operations.o \
               $(BUILD_DIR)/insert_reuse.o $(BUILD_DIR)/unfill.o \
               $(BUILD_DIR)/payload.o $(BUILD_DIR)/build_tree.o \
               $(BUILD_DIR)/delete_avail.o $(BUILD_DIR)/find.o
CPP_OBJECTS = $(BUILD_DIR)/graphGeneration.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/printUtils.o $(BUILD_DIR)/flatten.o

# Shared CUDA objects (everything except the entry-point .o files)
SHARED_CUDA_OBJECTS = $(BUILD_DIR)/HMotifCount.o $(BUILD_DIR)/HMotifCountUpdate.o $(BUILD_DIR)/operations.o \
                      $(BUILD_DIR)/insert_reuse.o $(BUILD_DIR)/unfill.o \
                      $(BUILD_DIR)/payload.o $(BUILD_DIR)/build_tree.o \
                      $(BUILD_DIR)/delete_avail.o $(BUILD_DIR)/find.o

# Default target
all: $(TARGET)

# Build all executables (main + type1 + type2 + type3)
all-types: $(TARGET) $(TARGET_TYPE1) $(TARGET_TYPE2) $(TARGET_TYPE3)

# Build the main executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Build type executables (share all objects except entry point)
$(TARGET_TYPE1): $(BUILD_DIR)/type1.o $(SHARED_CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_TYPE2): $(BUILD_DIR)/type2.o $(SHARED_CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_TYPE3): $(BUILD_DIR)/type3.o $(SHARED_CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile CUDA source files under structure/
$(BUILD_DIR)/%.o: $(STRUCT_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile CUDA source files under kernel/
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile utility source files
$(BUILD_DIR)/%.o: $(UTILS_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Run the program with default parameters (includes payload capacity)
run: $(TARGET)
	$(TARGET) 8 5 1 100 4096

# Run individual type executables
run-type1: $(TARGET_TYPE1)
	$(TARGET_TYPE1) 8 5 1 100 4096

run-type2: $(TARGET_TYPE2)
	$(TARGET_TYPE2) 8 5 1 100 4096

run-type3: $(TARGET_TYPE3)
	$(TARGET_TYPE3) 8 5 1 100 4096

# Run all executables sequentially (main + type1 + type2 + type3)
run-all: $(TARGET) $(TARGET_TYPE1) $(TARGET_TYPE2) $(TARGET_TYPE3)
	@echo "=== Running main (30-bin motif counts) ==="
	$(TARGET) 8 5 1 100 4096
	@echo ""
	@echo "=== Running Type1 motif ==="
	$(TARGET_TYPE1) 8 5 1 100 4096
	@echo ""
	@echo "=== Running Type2 motif ==="
	$(TARGET_TYPE2) 8 5 1 100 4096
	@echo ""
	@echo "=== Running Type3 motif ==="
	$(TARGET_TYPE3) 8 5 1 100 4096

# Run with custom parameters (usage: make run-custom ARGS="10 3 1 50 8192")
run-custom: $(TARGET)
	$(TARGET) $(ARGS)

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
	@echo "  all         - Build the main executable (default)"
	@echo "  all-types   - Build all executables (main + type1 + type2 + type3)"
	@echo "  run         - Build and run main with default parameters (8 5 1 100 4096)"
	@echo "  run-type1   - Build and run type1 motif counter"
	@echo "  run-type2   - Build and run type2 motif counter"
	@echo "  run-type3   - Build and run type3 motif counter"
	@echo "  run-all     - Build and run all executables sequentially"
	@echo "  run-custom  - Run with custom parameters (make run-custom ARGS=\"10 3 1 50 8192\")"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  install     - Show installation instructions"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make run                    # Run with default: 8 hyperedges, 5 max vertices, IDs 1-100, capacity 4096"
	@echo "  make run-type1              # Run type1 motif counter with defaults"
	@echo "  make run-all                # Run all motif counters sequentially"
	@echo "  make run-custom ARGS=\"10 3 1 50 8192\"  # Run with: 10 hyperedges, 3 max vertices, IDs 1-50, capacity 8192"
	@echo "  ./build/main 20 4 1 200 16384    # Direct execution with custom capacity"

# Phony targets
.PHONY: all all-types run run-type1 run-type2 run-type3 run-all run-custom clean rebuild install help
