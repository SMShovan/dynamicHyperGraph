#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

#include <vector>
#include <cuda_runtime.h>

// CBST node definition shared by kernels across translation units
struct CBSTNode {
    int index;
    int value;
    int length;
    int size;
    CBSTNode* left;
    CBSTNode* right;
    CBSTNode* parent;
};

// Helper available to both host/device where needed
__host__ __device__ static inline int nextMultipleOf4(int num) {
    if (num == 0) return 0;
    return ((num + 4) / 4) * 4;
}

// Function declarations for Complete Binary Search Tree operations
void constructCompleteBinarySearchTree(
    int* h_indices, int* h_values, int n, 
    int* flatValues, int flatValuesSize, 
    int* h_indices2, int* h_values2, 
    int* flatValues2, int flatValuesSize2, 
    int* h_indices3, int* h_values3, 
    int* flatValues3, int flatValuesSize3
);

// Helper functions for CBST operations
void checkCuda(cudaError_t result);
std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d);

#endif // STRUCTURE_HPP
