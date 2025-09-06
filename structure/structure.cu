#include "../include/structure.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <climits>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// Helper functions (declared inline in header for single-definition)
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

__device__ int floor_log2(int x) {
    int log = 0;
    while (x >>= 1) ++log;
    return log;
}

// CUDA Kernels
__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        nodes[tid].index = tid;
        nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
        nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
        nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    }
}

__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int log2_tid = floor_log2(tid + 1);
        int log2_n = floor_log2(n);
        int index =  ((2 * (tid + 1  - (1<<log2_tid))) + 1) * (1 << log2_n) / (1 << log2_tid);
        int index2 = min(index, index - (index/2) + (n + 1 - (1<< log2_n)));
        index2--;

        nodes[tid].size = totalSize;
        if (index2 < n) {
            nodes[tid].index = indices[index2];
            nodes[tid].value = values[index2];
            if (index2 < n - 1) {
                nodes[tid].length = values[index2 + 1] - values[index2];
            } else {
                nodes[tid].length = totalSize - values[index2];
            }
        }
    }
}

__global__ void printEachNode(CBSTNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= n) {
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != tid) {
            if (current->index > tid) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Length = %d, Size = %d\n",
                   tid, current->index, current->value, current->length, current->size);
        }
    }
}

// Main function implementation
void constructCompleteBinarySearchTree(int* h_indices, int* h_values, int n, int* flatValues, int flatValuesSize, int* h_indices2, int* h_values2, int* flatValues2, int flatValuesSize2, int* h_indices3, int* h_values3, int* flatValues3, int flatValuesSize3) {
    const int fixedSize = 1024; // Fixed size for d_flatValues

    // Check if fixedSize is at least flatValuesSize
    if (fixedSize < flatValuesSize) {
        std::cerr << "Overflow: fixedSize is less than flatValuesSize" << std::endl;
        return;
    }

    CBSTNode* d_nodes;
    int* d_indices;
    int* d_values;
    int* d_flatValues;
    int* d_insertIndices;
    int* d_insertValues;
    int* d_insertSizes;
    int* d_partialSolution;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_nodes, n * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&d_indices, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, n * sizeof(int)));

    // Allocate fixed memory for d_flatValues
    checkCuda(cudaMalloc(&d_flatValues, fixedSize * sizeof(int)));

    // Copy first portion from flatValues
    checkCuda(cudaMemcpy(d_flatValues, flatValues, flatValuesSize * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize remaining portion to zero
    checkCuda(cudaMemset(d_flatValues + flatValuesSize, 0, (fixedSize - flatValuesSize) * sizeof(int)));

    checkCuda(cudaMalloc(&d_insertIndices, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_insertValues, n * 3 * sizeof(int)));  // Allocate max size for values
    checkCuda(cudaMalloc(&d_insertSizes, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_partialSolution, 3 * n * sizeof(int)));

    checkCuda(cudaMemcpy(d_indices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy dummy insert indices and values for initial tree construction
    checkCuda(cudaMemcpy(d_insertIndices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues, h_values, n * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Build the empty binary tree
    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // Step 2: Store items into internal nodes
    storeItemsIntoNodes<<<numBlocks, blockSize>>>(d_nodes, d_indices, d_values, n, flatValuesSize);
    checkCuda(cudaDeviceSynchronize());

    // Step 3: Tree construction complete

    // Print each node from the device
    std::cout << "Printing the tree from the device:" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // For now, we'll just do basic tree construction
    // The full implementation with insertion/deletion would go here

    // Free device memory
    checkCuda(cudaFree(d_insertIndices));
    checkCuda(cudaFree(d_insertValues));
    checkCuda(cudaFree(d_insertSizes));
    checkCuda(cudaFree(d_indices));
    checkCuda(cudaFree(d_values));
    checkCuda(cudaFree(d_nodes));
    checkCuda(cudaFree(d_flatValues));
}
