#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <algorithm>
#include <set>
// Include Thrust headers
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__device__ int id_to_index[128] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    21, 23, 22, 24, 23, 25, 24, 26,
    0, 0, 0, 0, 0, 0, 0, 0,
    21, 22, 23, 24, 23, 24, 25, 26,
    21, 23, 23, 25, 22, 24, 24, 26,
    27, 28, 28, 29, 28, 29, 29, 30,
    1, 2, 2, 3, 2, 3, 3, 4,
    5, 6, 6, 8, 7, 9, 9, 10,
    5, 7, 6, 9, 6, 9, 8, 10,
    11, 13, 12, 14, 13, 15, 14, 16,
    5, 6, 7, 9, 6, 8, 9, 10,
    11, 12, 13, 14, 13, 14, 15, 16,
    11, 13, 13, 15, 12, 14, 14, 16,
    17, 18, 18, 19, 18, 19, 19, 20
};

// CUDA Kernel to compute motif index and count occurrences
__device__ void count_motif(int deg_a, int deg_b, int deg_c, int C_ab, int C_bc, int C_ca, int g_abc, int* motif_counts, int n, int idx) {


    int count = 0;

    int a = deg_a - (C_ab + C_ca) + g_abc;
    int b = deg_b - (C_bc + C_ab) + g_abc;
    int c = deg_c - (C_ca + C_bc) + g_abc;
    int d = C_ab - g_abc;
    int e = C_bc - g_abc;
    int f = C_ca - g_abc;
    int g = g_abc;

    int motif_id = (a > 0) + ((b > 0) << 1) + ((c > 0) << 2) + ((d > 0) << 3) + ((e > 0) << 4) + ((f > 0) << 5) + ((g > 0) << 6);
    int index = id_to_index[motif_id] - 1;


    // Store the count in the result array at index tid
    motif_counts[idx + index]++;
}



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

std::vector<std::vector<int>> alternate(const std::vector<std::vector<int>>& random2DVec) {
    // Step 1: Find the maximum value in random2DVec
    int maxValue = 0;
    for (const auto& row : random2DVec) {
        if (!row.empty()) {
            maxValue = std::max(maxValue, *std::max_element(row.begin(), row.end()));
        }
    }

    // Step 2: Initialize alter2DVec with size maxValue + 1 (to handle 0-indexing)
    std::vector<std::vector<int>> alter2DVec(maxValue + 1);

    // Step 3: Populate alter2DVec with indices from random2DVec
    for (int rowIndex = 0; rowIndex < random2DVec.size(); ++rowIndex) {
        for (int value : random2DVec[rowIndex]) {
            alter2DVec[value].push_back(rowIndex + 1);  // Insert the row index at the position of the value
        }
    }

    return alter2DVec;
}

std::vector<std::vector<int>> hyperedgeAdjacency(
    const std::vector<std::vector<int>>& vertexToHyperedge, 
    const std::vector<std::vector<int>>& hyperedgeToVertex) {
    
    int nHyperedges = hyperedgeToVertex.size();
    
    // Resultant adjacency matrix for hyperedges
    std::vector<std::vector<int>> hyperedgeAdjacencyMatrix(nHyperedges);

    // Iterate through each hyperedge
    for (int hyperedge = 0; hyperedge < nHyperedges; ++hyperedge) {
        std::set<int> adjacentHyperedges;

        // Get the vertices connected by this hyperedge
        const std::vector<int>& vertices = hyperedgeToVertex[hyperedge];

        // For each vertex, find other hyperedges connected to it
        for (int vertex : vertices) {
            for (int otherHyperedge : vertexToHyperedge[vertex]) {
                if (otherHyperedge != hyperedge + 1) {  // Avoid self-loop
                    adjacentHyperedges.insert(otherHyperedge); // Ensure no duplicates
                }
            }
        }

        // Convert set to vector and store in adjacency matrix
        hyperedgeAdjacencyMatrix[hyperedge] = std::vector<int>(adjacentHyperedges.begin(), adjacentHyperedges.end());
    }

    return hyperedgeAdjacencyMatrix;
}

__host__ __device__ int nextMultipleOf32(int num) {
    return ((num + 32) / 32) * 32;
}

__host__ __device__ int nextMultipleOf4(int num) {
    if (num == 0)
        return 0;
    return ((num + 4) / 4) * 4;
}
// Kernel to compute next multiple of 4 for each third element
__global__ void computeNextMultipleOf4(int* partialSolution, int* tmp, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
    {
        int val = partialSolution[3*idx + 2];
        tmp[idx] = nextMultipleOf4(val);
    }
}

// Kernel to update partialSolution with the prefix sum results
__global__ void updatePartialSolution(int* partialSolution, int* tmp, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
    {
        partialSolution[3*idx + 2] = tmp[idx];
    }
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

__global__ void findContents(RBTreeNode* nodes, int* searchIndices, int searchSize, int* flatValues) {
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

            int currLoc = current->value;
            printf("\n");
            while(flatValues[currLoc++] != INT_MIN)
            {
                printf("%d ", flatValues[currLoc]);
            }
            printf("\n");

            printf("Node %d: Index = %d, Value = %d, Length = %d, Color = %s\n", searchIndex, current->index, current->value, current->length, current->color ? "Black" : "Red");
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
                while (flatValues[valueIndex] != 0 && flatValues[valueIndex] != INT_MIN && flatValues[valueIndex] > 0) {
                    
                    // Needs to be tested
                    if (flatValues[valueIndex] < 0)
                    {
                        valueIndex = flatValues[valueIndex] * (-1);
                        continue;
                    }
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

__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize){
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

        int idxPartialSolution = tid * 3;
        int startPartialSolution = idxPartialSolution + 1;
        int lenPartialSolution = idxPartialSolution + 2;

        if (tid == 0)
            if (partialSolution[lenPartialSolution] == 0)
                return;
        else
            if (partialSolution[lenPartialSolution] == partialSolution[lenPartialSolution - 3] )
                return;
        
        int startIdx, endIdx;
        int storeStartIdx;
        if (tid == 0)
        {
            startIdx = spaceAvailableFrom;

        }
        else 
        {
            startIdx = spaceAvailableFrom + partialSolution[idxPartialSolution - 1];
        }

        storeStartIdx = startIdx;

        for (int i = partialSolution[startPartialSolution]; i < numValues; i++, startIdx++)
        {
            flatValues[startIdx] = values[i];
        }

        flatValues[storeStartIdx + partialSolution[lenPartialSolution] ] = INT_MIN;

        flatValues[partialSolution[idxPartialSolution]] = storeStartIdx * (-1);

        // # if __CUDA_ARCH__>=200
        //     printf("infinity set: %d with len %d \n", storeStartIdx + partialSolution[lenPartialSolution], partialSolution[lenPartialSolution] );
            
        // #endif

    }
}

void cumPartialSol(std::vector<int>& partialSolution){
    
    int cum = 0;
    for (int i = 0; i < partialSolution.size(); i++)
    {
        if ((i + 1) % 3 == 0)
        {
            partialSolution[i] = nextMultipleOf4(partialSolution[i]) + cum;
            cum = partialSolution[i];
        }
    }
}

__device__ int deg(int* d_h2vFlatvalues, int loc) {
    int count = 0;

    while (d_h2vFlatvalues[loc] != 0 && d_h2vFlatvalues[loc] != INT_MIN )
    {
        count++;
        loc++;
    }

    return count;
}

__device__ int con(int* d_h2vFlatvalues, int loc_a, int loc_b) {
    int count = 0;
    int i = loc_a;
    int j = loc_b;
    while (true) {
        // Terminate if any element is INT_MIN or 0
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0) {
            break;
        }
        
        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j]) {
            count++;  // Common item found
            i++;
            j++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j]) {
            i++;  // Move pointer in arr1
        } else {
            j++;  // Move pointer in arr2
        }

        
    }

    return count;
}

__device__ int group(int* d_h2vFlatvalues, int loc_a, int loc_b, int loc_c) {
    
    int i = loc_a, j = loc_b, k = loc_c;
    int count = 0;

    // Use a single loop with three pointers
    while (true) {
        // Terminate if any element is INT_MIN or 0 in any of the three arrays
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[k] == INT_MIN || 
            d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0 || d_h2vFlatvalues[k] == 0) {
            break;
        }

        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j] && d_h2vFlatvalues[j] == d_h2vFlatvalues[k]) {
            count++;  // Common item found in all three arrays
            i++;
            j++;
            k++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j] || d_h2vFlatvalues[i] < d_h2vFlatvalues[k]) {
            i++;  // Move pointer in arr1
        } else if (d_h2vFlatvalues[j] < d_h2vFlatvalues[i] || d_h2vFlatvalues[j] < d_h2vFlatvalues[k]) {
            j++;  // Move pointer in arr2
        } else {
            k++;  // Move pointer in arr3
        }

        
    }

    return count;
}

__global__ void updateCount(RBTreeNode * d_h2vNodes, int* d_h2vFlatvalues, 
                            RBTreeNode * d_v2hNodes, int* d_v2hFlatvalues, 
                            RBTreeNode * d_h2hNodes, int* d_h2hFlatvalues, int size, int * d_partialResults, int fixedSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
// Partial result startPointer
        int* startPointer = d_partialResults + idx * 30;
// Find the address of the starting node of the hyperedge idx
        int searchIndex = idx;
        RBTreeNode* id_a = d_h2vNodes;
        while (id_a != nullptr && id_a->index != searchIndex) {
            if (id_a->index > searchIndex) {
                id_a = id_a->left;
            } else {
                id_a = id_a->right;
            }
        }
        if (id_a != nullptr) {
            // printf("Node %d: Index = %d, Value = %d, Length = %d, Color = %s\n",
            //        searchIndex, current->index, current->value, current->length, current->color ? "Black" : "Red");

            int loc_a = id_a->value;
            

// Now search por adjacent hyperedge of a
            searchIndex = idx;
            RBTreeNode* id_b = d_h2vNodes;
            while (id_b != nullptr && id_b->index != searchIndex) {
                if (id_b->index > searchIndex) {
                    id_b = id_b->left;
                } else {
                    id_b = id_b->right;
                }
            }
            if (id_b != nullptr) {
                // printf("Node %d: Index = %d, Value = %d, Length = %d, Color = %s\n",
                //        searchIndex, current->index, current->value, current->length, current->color ? "Black" : "Red");

                int loc_b = id_b->value;
                
                int temp_loc_a = loc_a;
                int temp_loc_b = loc_b;

                while (true) {
                // Terminate if any element is INT_MIN or 0
                if (d_h2hFlatvalues[temp_loc_a] == INT_MIN || d_h2hFlatvalues[temp_loc_b] == INT_MIN || d_h2hFlatvalues[temp_loc_a] == 0 || d_h2hFlatvalues[temp_loc_b] == 0) {
                    break;
                }
                
                if (d_h2hFlatvalues[temp_loc_a] == d_h2hFlatvalues[temp_loc_b]) {
// Now process triangles 
                    searchIndex = d_h2hFlatvalues[temp_loc_a]; 

                    RBTreeNode* id_c = d_h2vNodes;
                    while (id_c != nullptr && id_c->index != searchIndex) {
                        if (id_c->index > searchIndex) {
                            id_c = id_c->left;
                        } else {
                            id_c = id_c->right;
                        }
                    }
                    if (id_c != nullptr) {
                        // printf("Node %d: Index = %d, Value = %d, Length = %d, Color = %s\n",
                        //        searchIndex, current->index, current->value, current->length, current->color ? "Black" : "Red");

                        int loc_c = id_c->value;

// Now get deg_(a,b,c), con_{(a,b),(b,c),(c,a)}, con_{(a,b,c)}
                        int deg_a = deg(d_h2vFlatvalues, loc_a);
                        int deg_b = deg(d_h2vFlatvalues, loc_b);
                        int deg_c = deg(d_h2vFlatvalues, loc_c);

                        int con_ab = con(d_h2vFlatvalues, loc_a, loc_b);
                        int con_bc = con(d_h2vFlatvalues, loc_b, loc_c);
                        int con_ca = con(d_h2vFlatvalues, loc_c, loc_a);

                        int g_abc = group(d_h2vFlatvalues, loc_a, loc_b, loc_c);

                        count_motif(deg_a, deg_b, deg_c, con_ab, con_bc, con_ca, g_abc, d_partialResults, 1, idx);

                    
                    }
                    else{
                        return;
                    }

                    
                    temp_loc_a++;
                    temp_loc_b++;
                } else if (d_h2hFlatvalues[temp_loc_a] < d_h2hFlatvalues[temp_loc_b]) {
                    temp_loc_a++;  // Move pointer in arr1
                } else {
                    temp_loc_b++;  // Move pointer in arr2
                }

                // Ensure we don't go out of bounds
                if (temp_loc_a >= fixedSize || temp_loc_b >= fixedSize) {
                    break;
                }
            }


            } else {
                return;
            }

        } else {
            return;
        }

        
    }
}


void constructRedBlackTree(int* h_indices, int* h_values, int n, int* flatValues, int flatValuesSize, int* h_indices2, int* h_values2, int* flatValues2, int flatValuesSize2, int* h_indices3, int* h_values3, int* flatValues3, int flatValuesSize3) {
    const int fixedSize = 1024; // Fixed size for d_flatValues

//hyperedge2node

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

    

    // Now perform cumPartialSol in parallel on device
    int K = insertIndices.size();
    int* d_tmp;
    checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));

    computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Perform inclusive scan over d_tmp using Thrust
    thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp);
    thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
    checkCuda(cudaDeviceSynchronize());

    // Update partialSolution[3*k+2] = tmp[k];
    updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Copy partialSolution back to host and print
    checkCuda(cudaMemcpy(partialSolution.data(), d_partialSolution, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(partialSolution, "Cumulative Partial solution");


    printf("Space available from: %d \n", flatValuesSize);

    allocateSpace<<<(insertIndices.size() + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution, d_flatValues, flatValuesSize, d_insertIndices, d_insertValues, d_insertSizes, insertIndices.size());
    checkCuda(cudaDeviceSynchronize());
    // Copy flat values back to host and print them
    std::vector<int> updatedFlatValues(fixedSize);
    checkCuda(cudaMemcpy(updatedFlatValues.data(), d_flatValues, fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(updatedFlatValues, "Updated Flattened Values (vec1d)");



//node2hyperedge
    // Check if fixedSize is at least flatValuesSize
    if (fixedSize < flatValuesSize2) {
        std::cerr << "Overflow: fixedSize is less than flatValuesSize2" << std::endl;
        return;
    }

    RBTreeNode* d_nodes2;
    int* d_indices2;
    int* d_values2;
    int* d_flatValues2;
    int* d_insertIndices2;
    int* d_insertValues2;
    int* d_insertSizes2;
    int* d_partialSolution2;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_nodes2, n * sizeof(RBTreeNode)));
    checkCuda(cudaMalloc(&d_indices2, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_values2, n * sizeof(int)));

    // Allocate fixed memory for d_flatValues
    checkCuda(cudaMalloc(&d_flatValues2, fixedSize * sizeof(int)));

    // Copy first portion from flatValues
    checkCuda(cudaMemcpy(d_flatValues2, flatValues2, flatValuesSize2 * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize remaining portion to zero
    checkCuda(cudaMemset(d_flatValues2 + flatValuesSize2, 0, (fixedSize - flatValuesSize2) * sizeof(int)));

    checkCuda(cudaMalloc(&d_insertIndices2, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_insertValues2, n * 3 * sizeof(int)));  // Allocate max size for values
    checkCuda(cudaMalloc(&d_insertSizes2, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_partialSolution2, 3 * n * sizeof(int)));

    checkCuda(cudaMemcpy(d_indices2, h_indices2, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values2, h_values2, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy dummy insert indices and values for initial tree construction
    checkCuda(cudaMemcpy(d_insertIndices2, h_indices2, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues2, h_values2, n * sizeof(int), cudaMemcpyHostToDevice));

    blockSize = 256;
    numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Build the empty binary tree
    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(d_nodes2, n);
    checkCuda(cudaDeviceSynchronize());

    // Step 2: Store items into internal nodes
    storeItemsIntoNodes<<<numBlocks, blockSize>>>(d_nodes2, d_indices2, d_values2, n, flatValuesSize2);
    checkCuda(cudaDeviceSynchronize());

    // Step 3: Color the nodes
    colorNodes<<<numBlocks, blockSize>>>(d_nodes2, n);
    checkCuda(cudaDeviceSynchronize());

    // Print each node from the device
    std::cout << "Printing the tree from the device:" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(d_nodes2, n);
    checkCuda(cudaDeviceSynchronize());

    // Prepare data for insertion
    std::vector<std::pair<int, std::vector<int>>> insertVector2 = {{2, {200 }}, {4, {400, 300, 310, 320, 330, 340, 350}}, {6, {600, 700, 650}}};
    std::vector<int> insertIndices2(insertVector2.size());
    std::vector<int> insertValues2;
    std::vector<int> insertSizes2(insertVector2.size());
    std::vector<int> partialSolution2(insertVector2.size() * 3, 0);
    

    for (size_t i = 0; i < insertVector2.size(); ++i) {
        insertIndices2[i] = insertVector2[i].first;
        insertValues2.insert(insertValues2.end(), insertVector2[i].second.begin(), insertVector2[i].second.end());
        if (i == 0)
            insertSizes2[i] = insertVector2[i].second.size();
        else 
            insertSizes2[i] = insertSizes2[i-1] + insertVector2[i].second.size();
    }

    checkCuda(cudaMemcpy(d_insertIndices2, insertIndices2.data(), insertIndices2.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues2, insertValues2.data(), insertValues2.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertSizes2, insertSizes2.data(), insertSizes2.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_partialSolution2, partialSolution2.data(), insertSizes2.size() * sizeof(int) * 3, cudaMemcpyHostToDevice));

    // Insert nodes into the Red-Black Tree
    insertNode<<<(insertIndices2.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes2, d_flatValues2, d_insertIndices2, d_insertValues2, d_insertSizes2, insertIndices2.size(), d_partialSolution2);
    checkCuda(cudaDeviceSynchronize());

    

    K = insertIndices2.size();
    checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));

    computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution2, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Perform inclusive scan over d_tmp using Thrust
    tmp_ptr = thrust::device_pointer_cast(d_tmp);
    thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
    checkCuda(cudaDeviceSynchronize());

    // Update partialSolution2[3*k+2] = tmp[k];
    updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution2, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Copy partialSolution2 back to host and print
    checkCuda(cudaMemcpy(partialSolution2.data(), d_partialSolution2, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(partialSolution2, "Cumulative Partial solution for nodes2");

    printf("Space available from: %d \n", flatValuesSize2);

    allocateSpace<<<(insertIndices2.size() + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution2, d_flatValues2, flatValuesSize2, d_insertIndices2, d_insertValues2, d_insertSizes2, insertIndices2.size());

    // Copy flat values back to host and print them
    std::vector<int> updatedFlatValues2(fixedSize);
    checkCuda(cudaMemcpy(updatedFlatValues2.data(), d_flatValues2, fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(updatedFlatValues2, "Updated Flattened Values (vec1d)");

//  hyperedge to hyperedge
    // Check if fixedSize is at least flatValuesSize
    if (fixedSize < flatValuesSize3) {
        std::cerr << "Overflow: fixedSize is less than flatValuesSize3" << std::endl;
        return;
    }

    RBTreeNode* d_nodes3;
    int* d_indices3;
    int* d_values3;
    int* d_flatValues3;
    int* d_insertIndices3;
    int* d_insertValues3;
    int* d_insertSizes3;
    int* d_partialSolution3;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_nodes3, n * sizeof(RBTreeNode)));
    checkCuda(cudaMalloc(&d_indices3, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_values3, n * sizeof(int)));

    // Allocate fixed memory for d_flatValues
    checkCuda(cudaMalloc(&d_flatValues3, fixedSize * sizeof(int)));

    // Copy first portion from flatValues
    checkCuda(cudaMemcpy(d_flatValues3, flatValues3, flatValuesSize3 * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize remaining portion to zero
    checkCuda(cudaMemset(d_flatValues3 + flatValuesSize3, 0, (fixedSize - flatValuesSize3) * sizeof(int)));

    checkCuda(cudaMalloc(&d_insertIndices3, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_insertValues3, n * 3 * sizeof(int)));  // Allocate max size for values
    checkCuda(cudaMalloc(&d_insertSizes3, n * sizeof(int)));
    checkCuda(cudaMalloc(&d_partialSolution3, 3 * n * sizeof(int)));

    checkCuda(cudaMemcpy(d_indices3, h_indices3, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values3, h_values3, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy dummy insert indices and values for initial tree construction
    checkCuda(cudaMemcpy(d_insertIndices3, h_indices3, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues3, h_values3, n * sizeof(int), cudaMemcpyHostToDevice));

    blockSize = 256;
    numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Build the empty binary tree
    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(d_nodes3, n);
    checkCuda(cudaDeviceSynchronize());

    // Step 2: Store items into internal nodes
    storeItemsIntoNodes<<<numBlocks, blockSize>>>(d_nodes3, d_indices3, d_values3, n, flatValuesSize3);
    checkCuda(cudaDeviceSynchronize());

    // Step 3: Color the nodes
    colorNodes<<<numBlocks, blockSize>>>(d_nodes3, n);
    checkCuda(cudaDeviceSynchronize());

    // Print each node from the device
    std::cout << "Printing the tree from the device:" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(d_nodes3, n);
    checkCuda(cudaDeviceSynchronize());

    // Prepare data for insertion
    std::vector<std::pair<int, std::vector<int>>> insertVector3 = {{2, {200 }}};
    std::vector<int> insertIndices3(insertVector3.size());
    std::vector<int> insertValues3;
    std::vector<int> insertSizes3(insertVector3.size());
    std::vector<int> partialSolution3(insertVector3.size() * 3, 0);
    

    for (size_t i = 0; i < insertVector3.size(); ++i) {
        insertIndices3[i] = insertVector3[i].first;
        insertValues3.insert(insertValues3.end(), insertVector3[i].second.begin(), insertVector3[i].second.end());
        if (i == 0)
            insertSizes3[i] = insertVector3[i].second.size();
        else 
            insertSizes3[i] = insertSizes3[i-1] + insertVector3[i].second.size();
    }

    checkCuda(cudaMemcpy(d_insertIndices3, insertIndices3.data(), insertIndices3.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertValues3, insertValues3.data(), insertValues3.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_insertSizes3, insertSizes3.data(), insertSizes3.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_partialSolution3, partialSolution3.data(), insertSizes3.size() * sizeof(int) * 3, cudaMemcpyHostToDevice));

    // Insert nodes into the Red-Black Tree
    insertNode<<<(insertIndices3.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes3, d_flatValues3, d_insertIndices3, d_insertValues3, d_insertSizes3, insertIndices3.size(), d_partialSolution3);
    checkCuda(cudaDeviceSynchronize());

    

    checkCuda(cudaMemcpy(partialSolution3.data(), d_partialSolution3, insertSizes3.size() * sizeof(int) * 3, cudaMemcpyDeviceToHost));
    printVector(partialSolution3, "Partial solution");
    

    K = insertIndices3.size();
    checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));

    computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution3, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Perform inclusive scan over d_tmp using Thrust
    tmp_ptr = thrust::device_pointer_cast(d_tmp);
    thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
    checkCuda(cudaDeviceSynchronize());

    // Update partialSolution3[3*k+2] = tmp[k];
    updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution3, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    // Copy partialSolution3 back to host and print
    checkCuda(cudaMemcpy(partialSolution3.data(), d_partialSolution3, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(partialSolution3, "Cumulative Partial solution for nodes3");

    printf("Space available from: %d \n", flatValuesSize3);

    allocateSpace<<<(insertIndices3.size() + blockSize - 1) / blockSize, blockSize>>>(d_partialSolution3, d_flatValues3, flatValuesSize3, d_insertIndices3, d_insertValues3, d_insertSizes3, insertIndices3.size());

    // Copy flat values back to host and print them
    std::vector<int> updatedFlatValues3(fixedSize);
    checkCuda(cudaMemcpy(updatedFlatValues3.data(), d_flatValues3, fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(updatedFlatValues3, "Updated Flattened Values (vec1d)");



// Now algorithm having d_flatValues, d_flatValues2, updatedFlatValues2, updatedFlatValues2
    std::vector<int> search = {1,2,3};
    int *d_search;
    checkCuda(cudaMalloc(&d_search, search.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_search, search.data(), search.size() * sizeof(int), cudaMemcpyHostToDevice ));
    findContents<<<(search.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes, d_search, search.size(), d_flatValues);
    checkCuda(cudaDeviceSynchronize());
    findContents<<<(search.size() + blockSize - 1) / blockSize, blockSize>>>(d_nodes2, d_search, search.size(), d_flatValues2);
    checkCuda(cudaDeviceSynchronize());

// Storage for partial result
    int m = 30; // Number of columns
    std::vector<std::vector<int>> partialResults(n, std::vector<int>(m));

    // Fill partialResults with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            partialResults[i][j] = i * m + j;
        }
    }

    // Step 1: Flatten the 2D vector into a 1D array
    std::vector<int> flatPartialResults(n * m);
    for (int i = 0; i < n; ++i) {
        std::copy(partialResults[i].begin(), partialResults[i].end(), flatPartialResults.begin() + i * m);
    }

    // Step 2: Allocate memory on the device
    int* d_partialResults;
    size_t size = n * m * sizeof(int);
    cudaMalloc(&d_partialResults, size);

    // Step 3: Copy the flattened data to the device
    cudaMemcpy(d_partialResults, flatPartialResults.data(), size, cudaMemcpyHostToDevice);

    

    updateCount<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_nodes, d_flatValues, d_nodes2, d_flatValues2, d_nodes3, d_flatValues3, n, d_partialResults, fixedSize);

    checkCuda(cudaDeviceSynchronize());


// Free device memory
    checkCuda(cudaFree(d_insertIndices));
    checkCuda(cudaFree(d_insertValues));
    checkCuda(cudaFree(d_insertSizes));
    checkCuda(cudaFree(d_indices));
    checkCuda(cudaFree(d_values));
    checkCuda(cudaFree(d_nodes));
    checkCuda(cudaFree(d_flatValues));

    checkCuda(cudaFree(d_insertIndices2));
    checkCuda(cudaFree(d_insertValues2));
    checkCuda(cudaFree(d_insertSizes2));
    checkCuda(cudaFree(d_indices2));
    checkCuda(cudaFree(d_values2));
    checkCuda(cudaFree(d_nodes2));
    checkCuda(cudaFree(d_flatValues2));
}

int main() {
    int n = 8;
    std::vector<std::vector<int>> random2DVec = createRandom2DVector(n, 5, 1, 100);
    std::vector<std::vector<int>> alter2DVec = alternate(random2DVec);
    std::cout<< "Hyperedge to vertex"<< std::endl;
    print2DVector(random2DVec);
    std::cout<< "Vertex to hyperedge"<< std::endl;
    print2DVector(alter2DVec);
    std::vector<std::vector<int>> h2h = hyperedgeAdjacency(alter2DVec, random2DVec);
    std::cout<< "Hyperedge to hyperedge"<< std::endl;
    print2DVector(h2h);


    // Flatten the 2D vector
    auto flattened = flatten2DVector(random2DVec);
    auto flattened2 = flatten2DVector(alter2DVec);
    auto flattened3 = flatten2DVector(h2h);

    std::vector<int> flatValues = flattened.first;
    std::vector<int> flatIndices = flattened.second;

    std::vector<int> flatValues2 = flattened2.first;
    std::vector<int> flatIndices2 = flattened2.second;

    std::vector<int> flatValues3 = flattened3.first;
    std::vector<int> flatIndices3 = flattened3.second;



    // Print the flattened vectors
    printVector(flatValues, "Flattened Values (vec1d)");
    printVector(flatIndices, "Flattened Indices (vec2dto1d)");

    printVector(flatValues2, "Flattened Values2 (vec1d)");
    printVector(flatIndices2, "Flattened Indices2 (vec2dto1d)");

    printVector(flatValues3, "Flattened Values3 (vec1d)");
    printVector(flatIndices3, "Flattened Indices3 (vec2dto1d)");



    int* h_values = flatIndices.data();
    int* h_indices = new int[flatIndices.size()];
    for (size_t i = 0; i < flatIndices.size(); ++i) {
        h_indices[i] = i + 1;
    }

    int* h_values2 = flatIndices2.data();
    int* h_indices2 = new int[flatIndices2.size()];
    for (size_t i = 0; i < flatIndices2.size(); ++i) {
        h_indices2[i] = i + 1;
    }

    int* h_values3 = flatIndices3.data();
    int* h_indices3 = new int[flatIndices3.size()];
    for (size_t i = 0; i < flatIndices3.size(); ++i) {
        h_indices3[i] = i + 1;
    }


    constructRedBlackTree(h_indices, h_values, n, flatValues.data(), flatValues.size(), h_indices2, h_values2, flatValues2.data(), flatValues2.size(),  h_indices3, h_values3, flatValues3.data(), flatValues3.size());

    //constructRedBlackTree(h_indices2, h_values2, n, flatValues2.data(), flatValues2.size());


    delete[] h_indices;
    delete[] h_indices2;
    delete[] h_indices3;
    return 0;
}