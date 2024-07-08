#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <climits>

//Function to create a random 2-D vector
std::vector<std::vector<int>> createRandom2DVector(int n, int m, int r1, int r2) {
    std::vector<std::vector<int>> vec2d(n);
    std::srand(std::time(0)); // Seed for random number generation

    for (int i = 0; i < n; ++i) {
        int innerSize = rand() % m + 1; // Random inner size from 1 to m
        vec2d[i].resize(innerSize);
        for (int j = 0; j < innerSize; ++j) {
            vec2d[i][j] = rand() % (r2 - r1 + 1) + r1; // Random value in range [r1, r2]
        }
    }

    return vec2d;
}

int nextMultipleOf32(int num) {
    return ((num + 31) / 32) * 32;
}

std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> vec1d;
    std::vector<int> vec2dto1d(vec2d.size());

    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        vec2dto1d[i] = index;
        int innerSize = vec2d[i].size();
        int paddedSize = nextMultipleOf32(innerSize);
        for (int j = 0; j < paddedSize; ++j) {
            if (j < innerSize) {
                vec1d.push_back(vec2d[i][j]);
            } else if (j == paddedSize - 1) {
                vec1d.push_back(INT_MIN); // Padding with negative infinity
            } else {
                vec1d.push_back(0); // Padding with zeros
            }
            ++index;
        }
    }

    return {vec1d, vec2dto1d};
}

void print2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::cout << "2D Vector (Matrix Form):" << std::endl;
    for (const auto& row : vec2d) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(const std::vector<int>& vec, const std::string& name) {
    std::cout << name << ": [ ";
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}


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


// Kernel to print each node from the device
__global__ void printEachNode(RBTreeNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= n) {
        RBTreeNode* current = nodes;
        while (current != nullptr && current->index != tid) {
            if (current->index > tid) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Color = %s\n",
                   tid, current->index, current->value, current->color ? "Black" : "Red");
        }
    }
}
// Kernel to find and print nodes in the tree
__global__ void findNode(RBTreeNode* nodes, int* searchIndices, int searchSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        RBTreeNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Color = %s\n",
                   searchIndex, current->index, current->value, current->color ? "Black" : "Red");
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

void constructRedBlackTree(int* h_indices, int* h_values, int n, int* flatValues, int flatValuesSize) {
    RBTreeNode* d_nodes;
    int* d_indices;
    int* d_values;
    int* d_flatValues;

    checkCuda(cudaMalloc(&d_nodes, n * sizeof(RBTreeNode)));
    checkCuda(cudaMalloc(&d_indices, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_flatValues, flatValuesSize * sizeof(int)));

    checkCuda(cudaMemcpy(d_indices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_flatValues, flatValues, flatValuesSize * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Build the empty binary tree
    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // Step 2: Store items into internal nodes
    storeItemsIntoNodes<<<numBlocks, blockSize>>>(d_nodes, d_indices, d_values, n);
    checkCuda(cudaDeviceSynchronize());

    // Step 3: Color the nodes
    colorNodes<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // Print each node from the device
    std::cout << "Printing the tree from the device:" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());


    std::vector<int> searchVector = {0, 3, 6, 9};
    int* d_searchVector;
    checkCuda(cudaMalloc(&d_searchVector, searchVector.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_searchVector, searchVector.data(), searchVector.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Search for the nodes in parallel
    std::cout << "Searching for nodes in the tree:" << std::endl;
    findNode<<<(searchVector.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes, d_searchVector, searchVector.size());
    checkCuda(cudaDeviceSynchronize());

    // Free device memory
    checkCuda(cudaFree(d_searchVector));
    checkCuda(cudaFree(d_indices));
    checkCuda(cudaFree(d_values));
    checkCuda(cudaFree(d_nodes));
    checkCuda(cudaFree(d_flatValues));

    
}




int main() {
    int n = 8;
    std::vector<std::vector<int>> random2DVec = createRandom2DVector(n, 5, 1, 100);
    

    print2DVector(random2DVec);

    // Flatten the 2D vector
    auto flattened = flatten2DVector(random2DVec);
    std::vector<int> flatValues = flattened.first;
    std::vector<int> flatIndices = flattened.second;

    // Print the flattened vectors
    printVector(flatValues, "Flattened Values (vec1d)");
    printVector(flatIndices, "Flattened Indices (vec2dto1d)");

    int* h_values = flatIndices.data();
    int* h_indices = new int[flatIndices.size()];
    for (size_t i = 0; i < flatIndices.size(); ++i) {
        h_indices[i] = i + 1;
    }

    // int h_indices[] = {1, 2, 3, 4, 5, 6, 7, 8}; // Example indices
    // int h_values[] = {10, 20, 30, 40, 50, 60, 70, 80}; // Example values

    constructRedBlackTree(h_indices, h_values, n, flatValues.data(), flatValues.size());

    return 0;
}
