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

// id_to_index now defined only in main TU to avoid device linking complexities
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





// Device helpers and moved kernels are provided via kernel headers
#include "../kernel/device_utils.cuh"
#include "../kernel/kernels.cuh"
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


// ceil_log2 and floor_log2 are provided by kernel/device_utils.cuh

// CBSTNode and CBSTContext are declared in structure.hpp

// Kernels included from kernel/*.cu via kernels.cuh

// constructCBST moved to structure/operations.cu

// insertCBST moved to structure/operations.cu

// deleteCBST moved to structure/operations.cu





// Moved delete/avail kernels are included via kernels.cuh




// Payload kernels included via kernels.cuh

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

// Keep device helpers here to satisfy calls from updateCount in this TU
__device__ int deg(int* d_h2vFlatvalues, int loc) {
    int count = 0;
    while (d_h2vFlatvalues[loc] != 0 && d_h2vFlatvalues[loc] != INT_MIN ) {
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
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0) {
            break;
        }
        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j]) {
            count++;
            i++;
            j++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j]) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}
__device__ int group(int* d_h2vFlatvalues, int loc_a, int loc_b, int loc_c) {
    int i = loc_a, j = loc_b, k = loc_c;
    int count = 0;
    while (true) {
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[k] == INT_MIN ||
            d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0 || d_h2vFlatvalues[k] == 0) {
            break;
        }
        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j] && d_h2vFlatvalues[j] == d_h2vFlatvalues[k]) {
            count++;
            i++;
            j++;
            k++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j] || d_h2vFlatvalues[i] < d_h2vFlatvalues[k]) {
            i++;
        } else if (d_h2vFlatvalues[j] < d_h2vFlatvalues[i] || d_h2vFlatvalues[j] < d_h2vFlatvalues[k]) {
            j++;
        } else {
            k++;
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