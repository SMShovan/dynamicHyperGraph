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
    return ((num + 32) / 32) * 32;
}

int nextMultipleOf4(int num) {
    return ((num + 4) / 4) * 4;
}

std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> vec1d;
    std::vector<int> vec2dto1d(vec2d.size());

    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        vec2dto1d[i] = index;
        int innerSize = vec2d[i].size();
        int paddedSize = nextMultipleOf4(innerSize);
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
    int length;
    int size;
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
__global__ void storeItemsIntoNodes(RBTreeNode* nodes, int* indices, int* values, int n, int totalSize) {
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
            printf("Node %d: Index = %d, Value = %d, Length = %d, Size = %d, Color = %s\n",
                   tid, current->index, current->value, current->length, current->size, current->color ? "Black" : "Red");
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
            printf("Node %d: Index = %d, Value = %d, Length = %d, Color = %s\n",
                   searchIndex, current->index, current->value, current->length, current->color ? "Black" : "Red");
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

__global__ void insertNode(RBTreeNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < insertSize) {
        int insertIndex = insertIndices[tid];
        int* values;
        int numValues; 
        if (tid == 0){
            values = insertValues;
            numValues = insertSizes[tid];
        }
        else{
            values = insertValues + insertSizes[tid - 1];
            numValues = insertSizes[tid] - insertSizes[tid - 1];
        }
        // Search for the node by index
        RBTreeNode* current = nodes;
        while (current != nullptr && current->index != insertIndex) {
            if (current->index > insertIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }

        // If node is found
        if (current != nullptr) {
            int valueIndex = current->value;
            
            // Navigate flatValues array to find the position to insert
            for (int i = 0; i < numValues; ++i) {
                bool isOverflow = false;
                while (flatValues[valueIndex] != 0 && flatValues[valueIndex] != INT_MIN) {
                    if (flatValues[valueIndex + 1] == INT_MIN)
                    {
                        # if __CUDA_ARCH__>=200
                            printf("Overflow of thread %d: position %d start %d of size %d \n", tid, valueIndex + 1, i, numValues - i);
                            partialSolution[tid * 3] = valueIndex + 1;
                            partialSolution[tid * 3 + 1] = i; 
                            partialSolution[tid * 3 + 2] = numValues - i; 
                        #endif
                        isOverflow = true;
                    }
                    if (isOverflow)
                    {
                        break;
                    }
                    valueIndex++;
                }
                // Insert the new value
                if (isOverflow)
                    break;
                if (flatValues[valueIndex] != INT_MIN)
                    flatValues[valueIndex] = values[i];
            }

            // Update the node's value to the new index
            current->value = valueIndex;
        }
    }
}

void constructRedBlackTree(int* h_indices, int* h_values, int n, int* flatValues, int flatValuesSize) {
    const int fixedSize = 1024; // Fixed size for d_flatValues

    // Check if fixedSize is at least flatValuesSize
    if (fixedSize < flatValuesSize) {
        std::cerr << "Overflow: fixedSize is less than flatValuesSize" << std::endl;
        return;
    }

    RBTreeNode* d_nodes;
    int* d_indices;
    int* d_values;
    int* d_flatValues;
    int* d_insertIndices;
    int* d_insertValues;
    int* d_insertSizes;
    int* d_partialSolution;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_nodes, n * sizeof(RBTreeNode)));
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

    // Step 3: Color the nodes
    colorNodes<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // Print each node from the device
    std::cout << "Printing the tree from the device:" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(d_nodes, n);
    checkCuda(cudaDeviceSynchronize());

    // Prepare data for insertion
    std::vector<std::pair<int, std::vector<int>>> insertVector = {{2, {200 }}, {4, {400, 300, 310, 320, 330, 340, 350}}, {6, {600, 700, 650}}};
    std::vector<int> insertIndices(insertVector.size());
    std::vector<int> insertValues;
    std::vector<int> insertSizes(insertVector.size());
    std::vector<int> partialSolution(insertVector.size() * 3, 0);
    
    int count = 0;
    for (size_t i = 0; i < insertVector.size(); ++i) {
        insertIndices[i] = insertVector[i].first;
        insertValues.insert(insertValues.end(), insertVector[i].second.begin(), insertVector[i].second.end());
        if (i == 0)
            insertSizes[i] = insertVector[i].second.size();
        else 
            insertSizes[i] = insertSizes[i-1] + insertVector[i].second.size();
    }

    checkCuda(cudaMemcpy(d_insertIndices, insertIndices.data(), insertIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues, insertValues.data(), insertValues.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertSizes, insertSizes.data(), insertSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_partialSolution, partialSolution.data(), insertSizes.size() * sizeof(int) * 3, cudaMemcpyHostToDevice));

    // Insert nodes into the Red-Black Tree
    insertNode<<<(insertIndices.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes, d_flatValues, d_insertIndices, d_insertValues, d_insertSizes, insertIndices.size(), d_partialSolution);
    checkCuda(cudaDeviceSynchronize());

    // Copy flat values back to host and print them
    std::vector<int> updatedFlatValues(fixedSize);
    checkCuda(cudaMemcpy(updatedFlatValues.data(), d_flatValues, fixedSize * sizeof(int), cudaMemcpyDeviceToHost));

    checkCuda(cudaMemcpy(partialSolution.data(), d_partialSolution, insertSizes.size() * sizeof(int) * 3, cudaMemcpyDeviceToHost));
    printVector(partialSolution, "Partial solution");
    printVector(updatedFlatValues, "Updated Flattened Values (vec1d)");

    // Free device memory
    checkCuda(cudaFree(d_insertIndices));
    checkCuda(cudaFree(d_insertValues));
    checkCuda(cudaFree(d_insertSizes));
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

    constructRedBlackTree(h_indices, h_values, n, flatValues.data(), flatValues.size());


    delete[] h_indices;
    return 0;
}