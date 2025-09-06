#include "../include/structure.hpp"
#include "../include/utils.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// Forward declare kernels defined in main.cu
__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n);
__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize);
__global__ void printEachNode(CBSTNode* nodes, int n);
__global__ void insertNode(CBSTNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution);
__global__ void computeNextMultipleOf4(int* partialSolution, int* tmp, int K);
__global__ void updatePartialSolution(int* partialSolution, int* tmp, int K);
__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize);
__global__ void deleteNode(CBSTNode* nodes, int* deleteIndices, int deleteSize);
__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues);

// Local CUDA error checker for this TU
static inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

void constructCBST(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, int payloadCapacity, const char* datasetName, CBSTContext& ctx) {
    ctx.fixedSize = payloadCapacity;
    ctx.numRecords = numRecords;
    ctx.initialPayloadSize = flatPayloadSize;
    ctx.datasetName = datasetName;

    if (ctx.fixedSize < flatPayloadSize) {
        std::cerr << "Overflow: fixedSize is less than flatPayloadSize" << std::endl;
        return;
    }

    checkCuda(cudaMalloc(&ctx.d_nodes, numRecords * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&ctx.d_keys, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_startOffsets, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_flatPayload, ctx.fixedSize * sizeof(int)));

    checkCuda(cudaMemcpy(ctx.d_keys, keys, numRecords * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_startOffsets, startOffsets, numRecords * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(ctx.d_flatPayload, flatPayload, flatPayloadSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(ctx.d_flatPayload + flatPayloadSize, 0, (ctx.fixedSize - flatPayloadSize) * sizeof(int)));

    checkCuda(cudaMalloc(&ctx.d_insertKeys, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPayload, numRecords * 3 * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_relocationPlan, 3 * numRecords * sizeof(int)));

    int blockSize = 256;
    int numBlocks = (numRecords + blockSize - 1) / blockSize;

    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
    checkCuda(cudaDeviceSynchronize());

    storeItemsIntoNodes<<<numBlocks, blockSize>>>(ctx.d_nodes, ctx.d_keys, ctx.d_startOffsets, numRecords, flatPayloadSize);
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Printing the tree from the device (" << datasetName << "):" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
    checkCuda(cudaDeviceSynchronize());
}

void insertCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx) {
    if (insertKeys.empty()) return;
    std::vector<int> relocationPlanHost(insertKeys.size() * 3, 0);

    checkCuda(cudaMemcpy(ctx.d_insertKeys, insertKeys.data(), insertKeys.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPayload, insertPayload.data(), insertPayload.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, insertPrefixSizes.data(), insertPrefixSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_relocationPlan, relocationPlanHost.data(), relocationPlanHost.size() * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (static_cast<int>(insertKeys.size()) + blockSize - 1) / blockSize;
    insertNode<<<numBlocks, blockSize>>>(ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, static_cast<int>(insertKeys.size()), ctx.d_relocationPlan);
    checkCuda(cudaDeviceSynchronize());

    int K = static_cast<int>(insertKeys.size());
    int* d_tmp;
    checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));
    computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());
    thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp);
    thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
    checkCuda(cudaDeviceSynchronize());
    updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    std::vector<int> relocationPlanHostOut(K * 3);
    checkCuda(cudaMemcpy(relocationPlanHostOut.data(), ctx.d_relocationPlan, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(relocationPlanHostOut, "Cumulative Relocation Plan");

    printf("[%s] Space available from: %d \n", ctx.datasetName, ctx.initialPayloadSize);
    allocateSpace<<<numBlocks, blockSize>>>(ctx.d_relocationPlan, ctx.d_flatPayload, ctx.initialPayloadSize, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, K);
    checkCuda(cudaDeviceSynchronize());

    std::vector<int> updatedFlat(ctx.fixedSize);
    checkCuda(cudaMemcpy(updatedFlat.data(), ctx.d_flatPayload, ctx.fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(updatedFlat, "Updated Flattened Values (vec1d)");

    checkCuda(cudaFree(d_tmp));
}

void deleteCBST(const std::vector<int>& deleteKeys, CBSTContext& ctx) {
    if (deleteKeys.empty()) return;
    int* d_deleteKeys;
    checkCuda(cudaMalloc(&d_deleteKeys, deleteKeys.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_deleteKeys, deleteKeys.data(), deleteKeys.size() * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (static_cast<int>(deleteKeys.size()) + blockSize - 1) / blockSize;
    deleteNode<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys, static_cast<int>(deleteKeys.size()));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(d_deleteKeys));
}

// CBSTOperations implementation
CBSTOperations::CBSTOperations(const char* datasetName, int payloadCapacity) {
    ctx_.datasetName = datasetName;
    ctx_.fixedSize = payloadCapacity;
}

CBSTOperations::~CBSTOperations() {
    if (ctx_.d_insertKeys)       checkCuda(cudaFree(ctx_.d_insertKeys));
    if (ctx_.d_insertPayload)    checkCuda(cudaFree(ctx_.d_insertPayload));
    if (ctx_.d_insertPrefixSizes)checkCuda(cudaFree(ctx_.d_insertPrefixSizes));
    if (ctx_.d_relocationPlan)   checkCuda(cudaFree(ctx_.d_relocationPlan));
    if (ctx_.d_keys)             checkCuda(cudaFree(ctx_.d_keys));
    if (ctx_.d_startOffsets)     checkCuda(cudaFree(ctx_.d_startOffsets));
    if (ctx_.d_nodes)            checkCuda(cudaFree(ctx_.d_nodes));
    if (ctx_.d_flatPayload)      checkCuda(cudaFree(ctx_.d_flatPayload));
}

CBSTOperations::CBSTOperations(CBSTOperations&& other) noexcept { ctx_ = other.ctx_; constructed_ = other.constructed_; other = CBSTOperations(nullptr, 0); }
CBSTOperations& CBSTOperations::operator=(CBSTOperations&& other) noexcept { if (this != &other) { this->~CBSTOperations(); ctx_ = other.ctx_; constructed_ = other.constructed_; other = CBSTOperations(nullptr, 0); } return *this; }

void CBSTOperations::construct(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize) {
    constructCBST(keys, startOffsets, numRecords, flatPayload, flatPayloadSize, ctx_.fixedSize, ctx_.datasetName, ctx_);
    constructed_ = true;
}

void CBSTOperations::insert(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes) {
    insertCBST(insertKeys, insertPayload, insertPrefixSizes, ctx_);
}

void CBSTOperations::erase(const std::vector<int>& deleteKeys) {
    deleteCBST(deleteKeys, ctx_);
}

void CBSTOperations::findAndPrint(const std::vector<int>& ids) const {
    if (ids.empty()) return;
    int *d_search;
    checkCuda(cudaMalloc(&d_search, ids.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_search, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    findContents<<<(ids.size() + 256 - 1) / 256, 256>>>(ctx_.d_nodes, d_search, ids.size(), ctx_.d_flatPayload);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_search));
}


