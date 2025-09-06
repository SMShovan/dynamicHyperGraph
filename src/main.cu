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
// Include our utility functions
#include "../include/utils.hpp"
#include "../include/printUtils.hpp"
#include "../include/structure.hpp"

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


    // int count = 0; // Unused variable

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

// CBSTNode and CBSTContext are declared in structure.hpp

// Forward declarations for kernels used below
__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n);
__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize);
__global__ void printEachNode(CBSTNode* nodes, int n);
__global__ void insertNode(CBSTNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution);
__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize);
__global__ void deleteNode(CBSTNode* nodes, int* deleteIndices, int deleteSize);
__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues);

// constructCBST moved to structure/operations.cu

// insertCBST moved to structure/operations.cu

// deleteCBST moved to structure/operations.cu

// Kernel to build an empty binary tree
__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        nodes[tid].index = tid;
        nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
        nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
        nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    }
}

// Kernel to store items into internal nodes
__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize) {
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



// Kernel to print each node from the device
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
// Kernel to find and print nodes in the tree
__global__ void findNode(CBSTNode* nodes, int* searchIndices, int searchSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Length = %d\n",
                   searchIndex, current->index, current->value, current->length);
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
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

            printf("Node %d: Index = %d, Value = %d, Length = %d\n", searchIndex, current->index, current->value, current->length);
        } else {
            
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

// Mark avail = 1 for deleted keys
__global__ void markAvail(CBSTNode* nodes, int* deleteKeys, int deleteSize, int* avail) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < deleteSize) {
        int key = deleteKeys[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != key) {
            if (current->index > key) current = current->left; else current = current->right;
        }
        if (current != nullptr) {
            // current->index matches key; compute its array index by pointer arithmetic
            // Assuming nodes is a contiguous array
            int idx = static_cast<int>(current - nodes);
            avail[idx] = 1;
        }
    }
}

// Reduce avail bottom-up for [levelStart..levelEnd] range
__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords, int* avail, int* subtreeAvail) {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = levelStart + g;
    if (idx < 0 || idx > levelEnd || idx >= numRecords) return;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    int leftSum = (left < numRecords) ? subtreeAvail[left] : 0;
    int rightSum = (right < numRecords) ? subtreeAvail[right] : 0;
    subtreeAvail[idx] = avail[idx] + leftSum + rightSum;
}

__global__ void insertNode(CBSTNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution) {
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
        CBSTNode* current = nodes;
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

__global__ void deleteNode(
    CBSTNode* nodes,
    int* deleteIndices,
    int deleteSize
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < deleteSize) {
        int deleteIndex = deleteIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != deleteIndex) {
            if (current->index > deleteIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            // Simple deletion logic - mark node as deleted by setting index to -1
            current->index = -1;
            current = current->parent;
        }
    }
}


__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < insertSize) {
        // int insertIndex = insertIndices[tid]; // Unused variable
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
        
        int startIdx; // int endIdx; // Unused variable
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

__global__ void updateCount(CBSTNode * d_h2vNodes, int* d_h2vFlatvalues, 
                            CBSTNode * d_v2hNodes, int* d_v2hFlatvalues, 
                            CBSTNode * d_h2hNodes, int* d_h2hFlatvalues, int size, int * d_partialResults, int fixedSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
// Partial result startPointer
        // int* startPointer = d_partialResults + idx * 30; // Unused variable
// Find the address of the starting node of the hyperedge idx
        int searchIndex = idx;
        CBSTNode* id_a = d_h2vNodes;
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
            CBSTNode* id_b = d_h2vNodes;
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

                    CBSTNode* id_c = d_h2vNodes;
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


// operations moved to structure/operations.cu


int main(int argc, char* argv[]) {
    // Parse command line arguments
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    
    // Print parameters
    printHypergraphParams(params);
    
    // Generate hypergraph mappings
    auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);
    
    // Generate hyperedge-to-hyperedge adjacency
    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);
    std::cout << "Hyperedge to hyperedge" << std::endl;
    print2DVector(hyperedge2hyperedge);

    // Flatten the 2D vectors for GPU processing
    auto [h2vFlatVertexIds, h2vStartOffsets] = flatten(hyperedgeToVertex, "Hyperedge to Vertex");
    auto [v2hFlatHyperedgeIds, v2hStartOffsets] = flatten(vertexToHyperedge, "Vertex to Hyperedge");
    auto [h2hFlatAdjacency, h2hStartOffsets] = flatten(hyperedge2hyperedge, "Hyperedge to Hyperedge");

    // Prepare data for Complete Binary Search Tree construction
    auto [cbstH2VStartOffsets, cbstH2VKeys] = prepareCBSTData(h2vStartOffsets);
    auto [cbstV2HStartOffsets, cbstV2HKeys] = prepareCBSTData(v2hStartOffsets);
    auto [cbstH2HStartOffsets, cbstH2HKeys] = prepareCBSTData(h2hStartOffsets);

    // Construct Complete Binary Search Trees (generic) for each dataset using OO wrapper
    int numVertices = static_cast<int>(vertexToHyperedge.size());

    {
        CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
        h2vOps.construct(cbstH2VKeys, cbstH2VStartOffsets, params.numHyperedges,
                         h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));

        // Demo ops (optional)
        std::vector<std::pair<int, std::vector<int>>> insertVector = {{2, {200 }}, {4, {400, 300, 310, 320, 330, 340, 350}}, {6, {600, 700, 650}}};
        std::vector<int> demoInsertKeys;
        std::vector<int> demoInsertPayload;
        std::vector<int> demoInsertPrefixSizes(insertVector.size());
        demoInsertKeys.reserve(insertVector.size());
        for (size_t i = 0; i < insertVector.size(); ++i) {
            demoInsertKeys.push_back(insertVector[i].first);
            demoInsertPayload.insert(demoInsertPayload.end(), insertVector[i].second.begin(), insertVector[i].second.end());
            demoInsertPrefixSizes[i] = (i == 0) ? static_cast<int>(insertVector[i].second.size())
                                                : demoInsertPrefixSizes[i-1] + static_cast<int>(insertVector[i].second.size());
        }
        h2vOps.insert(demoInsertKeys, demoInsertPayload, demoInsertPrefixSizes);
        h2vOps.erase(std::vector<int>{2,4,6});
        h2vOps.findAndPrint(std::vector<int>{1,2,3});
    }

    {
        CBSTOperations v2hOps("V2H", params.payloadCapacity, params.alignment);
        v2hOps.construct(cbstV2HKeys, cbstV2HStartOffsets, numVertices,
                         v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
        v2hOps.findAndPrint(std::vector<int>{1,2,3});
    }

    {
        CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
        h2hOps.construct(cbstH2HKeys, cbstH2HStartOffsets, params.numHyperedges,
                         h2hFlatAdjacency.data(), static_cast<int>(h2hFlatAdjacency.size()));
        h2hOps.findAndPrint(std::vector<int>{1,2,3});
    }

    // Clean up memory
    delete[] cbstH2VKeys;
    delete[] cbstV2HKeys;
    delete[] cbstH2HKeys;
    
    return 0;
}