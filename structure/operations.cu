#include "../include/structure.hpp"
#include "../include/utils.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <climits>
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
// Propagation kernels
__global__ void markAvail(CBSTNode* nodes, int* deleteKeys, int deleteSize, int* avail);
__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords, int* avail, int* subtreeAvail);
// New kernel: place i-th insert into i-th deleted node
__global__ void insertIntoDeletedKth(CBSTNode* nodes,
                                     int* flatValues,
                                     int* subtreeAvail,
                                     int* avail,
                                     int numRecords,
                                     int* newKeys,
                                     int* newPayload,
                                     int* newPrefixSizes,
                                     int K) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= K) return;
    int k = tid + 1; // 1-based order statistic
    int idx = 0;
    while (idx < numRecords) {
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        int leftCount = (left < numRecords) ? subtreeAvail[left] : 0;
        int self = avail[idx];
        if (k <= leftCount) {
            idx = left;
            continue;
        }
        if (self == 1 && k == leftCount + 1) {
            break; // found deleted node at idx
        }
        k -= leftCount + self;
        idx = right;
    }
    // idx points to the target deleted node
    CBSTNode* node = &nodes[idx];
    int key = newKeys[tid];
    int start = (tid == 0) ? 0 : newPrefixSizes[tid - 1];
    int end = newPrefixSizes[tid];
    int len = end - start;
    int base = node->value;
    // write payload (assumption: len fits)
    for (int i = 0; i < len; ++i) {
        flatValues[base + i] = newPayload[start + i];
    }
    flatValues[base + len] = INT_MIN;
    // mark node re-used with new key and clear availability
    node->index = key;
    avail[idx] = 0;
}

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
    // ctx.alignment should be set by the owner (CBSTOperations) before calling

    if (numRecords == 0) {
        return;
    }

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
    // Availability arrays (0/1 per node) and subtree sums (per node)
    checkCuda(cudaMalloc(&ctx.d_avail, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_subtreeAvail, numRecords * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_avail, 0, numRecords * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, numRecords * sizeof(int)));

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

void fillCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx) {
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
    // Note: currently kernel pads to 4. To support arbitrary alignment, update the kernel to accept ctx.alignment.
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

    // Advance appended-space cursor so next batch appends after this one
    if (K > 0) {
        int totalAppended = relocationPlanHostOut[3 * (K - 1) + 2];
        ctx.initialPayloadSize += totalAppended;
    }

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
    // Mark node index = -1 (lazy delete)
    deleteNode<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys, static_cast<int>(deleteKeys.size()));
    checkCuda(cudaDeviceSynchronize());

    // Mark avail = 1 for deleted nodes
    markAvail<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys, static_cast<int>(deleteKeys.size()), ctx.d_avail);
    checkCuda(cudaDeviceSynchronize());

    // Bottom-up level-wise reduction
    // Compute last level start using heap property
    int lastLevelStart = 1;
    while (lastLevelStart * 2 <= ctx.numRecords) lastLevelStart <<= 1; // 2^h
    int levelStart = lastLevelStart - 1; // 0-based index of first node at last full level
    if (levelStart >= ctx.numRecords) levelStart = (lastLevelStart >> 1) - 1; // adjust if beyond size

    // Initialize subtreeAvail = avail at leaves and beyond
    // For ranges beyond numRecords, threads will just skip
    // Propagate from bottom to top
    int currStart = ctx.numRecords - 1; // last index
    // First, copy avail to subtreeAvail for all nodes
    int nodesBlocks = (ctx.numRecords + blockSize - 1) / blockSize;
    reduceAvailLevel<<<nodesBlocks, blockSize>>>(ctx.numRecords - 1, ctx.numRecords - 1, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());

    // Now perform level-wise reduction
    for (int levelEnd = ctx.numRecords - 1; levelStart >= 0; levelStart = (levelStart - 1) / 2) {
        int start = levelStart;
        int end = levelEnd;
        int count = end - start + 1;
        int blocks = (count + blockSize - 1) / blockSize;
        reduceAvailLevel<<<blocks, blockSize>>>(start, end, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
        checkCuda(cudaDeviceSynchronize());
        if (levelStart == 0) break;
        levelEnd = levelStart - 1;
        levelStart = (levelStart - 1) / 2;
    }

    checkCuda(cudaFree(d_deleteKeys));
}

void insertCBST(const std::vector<int>& newKeys,
                const std::vector<int>& newPayload,
                const std::vector<int>& newPrefixSizes,
                CBSTContext& ctx) {
    if (newKeys.empty()) return;
    // Copy inputs to device (reuse insert buffers)
    checkCuda(cudaMemcpy(ctx.d_insertKeys, newKeys.data(), newKeys.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPayload, newPayload.data(), newPayload.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, newPrefixSizes.data(), newPrefixSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    // Safety: ensure enough deletions exist (optional host-side check)
    // Launch K-thread kernel to place i-th insert into i-th deleted slot
    int K = static_cast<int>(newKeys.size());
    int blockSize = 256;
    int numBlocks = (K + blockSize - 1) / blockSize;
    insertIntoDeletedKth<<<numBlocks, blockSize>>>(ctx.d_nodes,
                                                   ctx.d_flatPayload,
                                                   ctx.d_subtreeAvail,
                                                   ctx.d_avail,
                                                   ctx.numRecords,
                                                   ctx.d_insertKeys,
                                                   ctx.d_insertPayload,
                                                   ctx.d_insertPrefixSizes,
                                                   K);
    checkCuda(cudaDeviceSynchronize());
    // Recompute subtreeAvail bottom-up to reflect consumed deletions
    int blockNodes = (ctx.numRecords + blockSize - 1) / blockSize;
    reduceAvailLevel<<<blockNodes, blockSize>>>(ctx.numRecords - 1, ctx.numRecords - 1, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());
    int lastLevelStart = 1;
    while (lastLevelStart * 2 <= ctx.numRecords) lastLevelStart <<= 1;
    int levelStart = lastLevelStart - 1;
    if (levelStart >= ctx.numRecords) levelStart = (lastLevelStart >> 1) - 1;
    for (int levelEnd = ctx.numRecords - 1; levelStart >= 0; levelStart = (levelStart - 1) / 2) {
        int start = levelStart;
        int end = levelEnd;
        int count = end - start + 1;
        int blocks = (count + blockSize - 1) / blockSize;
        reduceAvailLevel<<<blocks, blockSize>>>(start, end, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
        checkCuda(cudaDeviceSynchronize());
        if (levelStart == 0) break;
        levelEnd = levelStart - 1;
        levelStart = (levelStart - 1) / 2;
    }
}

// CBSTOperations implementation
CBSTOperations::CBSTOperations(const char* datasetName, int payloadCapacity, int alignment) {
    ctx_.datasetName = datasetName;
    ctx_.fixedSize = payloadCapacity;
    ctx_.alignment = alignment;
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

CBSTOperations::CBSTOperations(CBSTOperations&& other) noexcept {
    ctx_ = other.ctx_;
    constructed_ = other.constructed_;
    // Null out other's pointers to avoid double free
    other.ctx_.d_nodes = nullptr;
    other.ctx_.d_keys = nullptr;
    other.ctx_.d_startOffsets = nullptr;
    other.ctx_.d_flatPayload = nullptr;
    other.ctx_.d_insertKeys = nullptr;
    other.ctx_.d_insertPayload = nullptr;
    other.ctx_.d_insertPrefixSizes = nullptr;
    other.ctx_.d_relocationPlan = nullptr;
    other.constructed_ = false;
}

CBSTOperations& CBSTOperations::operator=(CBSTOperations&& other) noexcept {
    if (this != &other) {
        // Free current resources
        this->~CBSTOperations();
        // Steal other's resources
        ctx_ = other.ctx_;
        constructed_ = other.constructed_;
        // Null out other's pointers
        other.ctx_.d_nodes = nullptr;
        other.ctx_.d_keys = nullptr;
        other.ctx_.d_startOffsets = nullptr;
        other.ctx_.d_flatPayload = nullptr;
        other.ctx_.d_insertKeys = nullptr;
        other.ctx_.d_insertPayload = nullptr;
        other.ctx_.d_insertPrefixSizes = nullptr;
        other.ctx_.d_relocationPlan = nullptr;
        other.constructed_ = false;
    }
    return *this;
}

void CBSTOperations::construct(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize) {
    constructCBST(keys, startOffsets, numRecords, flatPayload, flatPayloadSize, ctx_.fixedSize, ctx_.datasetName, ctx_);
    constructed_ = true;
}

void CBSTOperations::insert(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes) {
    fillCBST(insertKeys, insertPayload, insertPrefixSizes, ctx_);
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


