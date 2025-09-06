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

// Device-side context is opaque; host context holds device buffers
struct CBSTContext {
    CBSTNode* d_nodes;
    int* d_keys;
    int* d_startOffsets;
    int* d_flatPayload;
    int* d_insertKeys;
    int* d_insertPayload;
    int* d_insertPrefixSizes;
    int* d_relocationPlan;
    int fixedSize;
    int numRecords;
    int initialPayloadSize;
    const char* datasetName;
};

// Host API for CBST operations
void constructCBST(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, const char* datasetName, CBSTContext& ctx);
void insertCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx);
void deleteCBST(const std::vector<int>& deleteKeys, CBSTContext& ctx);
void operations(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, const char* datasetName);

#endif // STRUCTURE_HPP
