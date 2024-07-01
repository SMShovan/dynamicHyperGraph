#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__device__ int ceil_log2(int x) {
    int log = 0;
    while ((1 << log) < x) ++log;
    return log;
}
__device__ int floor_log2(int x) {
    int log = 0;
    while (x >>= 1) ++log;
    return log;
}

// Structure for Red-Black Tree Node
struct RBTreeNode {
    int index;
    int value;
    bool color; // Red or Black
    RBTreeNode* left;
    RBTreeNode* right;
    RBTreeNode* parent;
};

// Kernel to build an empty binary tree
__global__ void buildEmptyBinaryTree(RBTreeNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        nodes[tid].index = tid;
        nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
        nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
        nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    }
}

// Kernel to store items into internal nodes
__global__ void storeItemsIntoNodes(RBTreeNode* nodes, int* indices, int* values, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int log2_tid = floor_log2(tid + 1);
        int log2_n = floor_log2(n);
        int index =  ((2 * (tid + 1  - (1<<log2_tid))) + 1) * (1 << log2_n) / (1 << log2_tid);
        int index2 = min(index, index - (index/2) + (n + 1 - (1<< log2_n)));
        index2--;


        // # if __CUDA_ARCH__>=200
        //     printf("tid is %d \n", tid + 1);
        //     printf("J(i) is %d \n", (tid + 1  - (1<<log2_tid)));
        //     printf("log2_n is %d \n", log2_n);
        //     printf("index is %d \n", index);
        //     printf("size is %d \n", n);
        // #endif

        // Ensure index is within bounds
        if (index2 - 1 < n) {
            nodes[tid].index = indices[index2];
            nodes[tid].value = values[index2];
        }
    }
}

// Kernel to color the nodes red or black
__global__ void colorNodes(RBTreeNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        nodes[tid].color = (tid % 2 == 0); // Simplified coloring: alternating red (false) and black (true)
    }
}


void constructRedBlackTree(int* h_indices, int* h_values, int n) {
    RBTreeNode* d_nodes;
    int* d_indices;
    int* d_values;

    cudaMalloc(&d_nodes, n * sizeof(RBTreeNode));
    cudaMalloc(&d_indices, n * sizeof(int));
    cudaMalloc(&d_values, n * sizeof(int));

    cudaMemcpy(d_indices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Build the empty binary tree
    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(d_nodes, n);
    cudaDeviceSynchronize();

    // Step 2: Store items into internal nodes
    storeItemsIntoNodes<<<numBlocks, blockSize>>>(d_nodes, d_indices, d_values, n);
    cudaDeviceSynchronize();

    // Step 3: Color the nodes
    colorNodes<<<numBlocks, blockSize>>>(d_nodes, n);
    cudaDeviceSynchronize();

    // Transfer the tree back to the host
    RBTreeNode* h_nodes = new RBTreeNode[n];
    cudaMemcpy(h_nodes, d_nodes, n * sizeof(RBTreeNode), cudaMemcpyDeviceToHost);

    std::cout << "Nodes after synchronization:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Node " << i << ": Index = " << h_nodes[i].index
        << ", Value = " << h_nodes[i].value
        << ", Color = " << (h_nodes[i].color ? "Black" : "Red") << std::endl;
    }

    // Free device memory
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_nodes);

    delete[] h_nodes;
}

int main() {
    int n = 8;
    int h_indices[] = {1, 2, 3, 4, 5, 6, 7, 8}; // Example indices
    int h_values[] = {10, 20, 30, 40, 50, 60, 70, 80}; // Example values

    constructRedBlackTree(h_indices, h_values, n);

    return 0;
}
